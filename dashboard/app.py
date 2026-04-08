"""Tiny Flask dashboard for the GA trading bot.

Reads three log files written by the bot:
    logs/heartbeat.json       — current trader state (atomic JSON)
    logs/paper_journal.jsonl  — append-only event journal (with rotation)
    logs/training.jsonl       — per-generation GA training progress

The dashboard is read-only — it never touches the bot or its model. Run
it as a separate process (or systemd unit) on the same Ubuntu host:

    python -m dashboard.app --host 0.0.0.0 --port 8080

Then open http://<public-ip>:8080 in a browser.

Optional auth: set DASHBOARD_TOKEN in the environment and pass it as
?token=... in the URL (or as the X-Dashboard-Token header). The repo
is private, but the public IP is reachable by anyone — turn this on
in any production-ish setup.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, abort, jsonify, render_template, request

from ga_bot.config import CONFIG, LOGS_DIR

HEARTBEAT_PATH = LOGS_DIR / "heartbeat.json"
JOURNAL_PATH = LOGS_DIR / "paper_journal.jsonl"
TRAINING_PATH = LOGS_DIR / "training.jsonl"

app = Flask(__name__)


# ---------- helpers ----------
def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_jsonl_tail(path: Path, max_lines: int = 500) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            buf = deque(f, maxlen=max_lines)
    except OSError:
        return []
    rows: List[Dict[str, Any]] = []
    for line in buf:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _heartbeat_age_seconds() -> float | None:
    hb = _read_json(HEARTBEAT_PATH)
    ts = hb.get("ts")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds()


def _build_equity_curve(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Take journal events and emit a list of {ts, balance} points."""
    points: List[Dict[str, Any]] = []
    for ev in events:
        if ev.get("event") in ("close", "sltp_close", "connect", "disconnect"):
            bal = ev.get("balance")
            ts = ev.get("ts")
            if bal is not None and ts:
                points.append({"ts": ts, "balance": float(bal)})
    return points


def _summarize_trades(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    closed = [e for e in events if e.get("event") in ("close", "sltp_close")]
    if not closed:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "net_pnl": 0.0, "best": 0.0, "worst": 0.0}
    pnls = [float(e.get("pnl", 0.0)) for e in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(closed) if closed else 0.0,
        "net_pnl": sum(pnls),
        "best": max(pnls) if pnls else 0.0,
        "worst": min(pnls) if pnls else 0.0,
    }


# ---------- auth ----------
def _check_token() -> None:
    expected = os.environ.get("DASHBOARD_TOKEN")
    if not expected:
        return  # auth disabled
    given = request.args.get("token") or request.headers.get("X-Dashboard-Token")
    if given != expected:
        abort(401)


@app.before_request
def _auth_gate() -> None:
    if request.path.startswith("/api/") or request.path == "/":
        _check_token()


# ---------- routes ----------
@app.route("/")
def index():
    return render_template("index.html", symbol=CONFIG.instrument.symbol)


@app.route("/api/status")
def api_status():
    hb = _read_json(HEARTBEAT_PATH)
    age = _heartbeat_age_seconds()
    journal = _read_jsonl_tail(JOURNAL_PATH, max_lines=500)
    summary = _summarize_trades(journal)

    training = _read_jsonl_tail(TRAINING_PATH, max_lines=500)
    training_active = False
    last_training = None
    if training:
        last_training = training[-1]
        training_active = last_training.get("event") != "done"

    return jsonify({
        "now": datetime.now(timezone.utc).isoformat(),
        "symbol": CONFIG.instrument.symbol,
        "leverage": CONFIG.account.leverage,
        "starting_balance": CONFIG.account.starting_balance,
        "heartbeat": hb,
        "heartbeat_age_sec": age,
        "trade_summary": summary,
        "training_active": training_active,
        "training_last_event": last_training,
    })


@app.route("/api/equity")
def api_equity():
    journal = _read_jsonl_tail(JOURNAL_PATH, max_lines=2000)
    return jsonify(_build_equity_curve(journal))


@app.route("/api/journal")
def api_journal():
    limit = int(request.args.get("limit", 50))
    rows = _read_jsonl_tail(JOURNAL_PATH, max_lines=max(limit, 1))
    return jsonify(list(reversed(rows))[:limit])


@app.route("/api/training")
def api_training():
    rows = _read_jsonl_tail(TRAINING_PATH, max_lines=2000)
    # Filter only per-generation rows for the chart.
    gens = [r for r in rows if "generation" in r]
    meta = next((r for r in rows if r.get("event") == "start"), None)
    done = next((r for r in reversed(rows) if r.get("event") == "done"), None)
    return jsonify({"meta": meta, "done": done, "generations": gens})


# ---------- entry ----------
def main() -> int:
    parser = argparse.ArgumentParser(description="GA bot dashboard server")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (0.0.0.0 makes it reachable on the public IP)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Dashboard listening on http://{args.host}:{args.port}")
    if os.environ.get("DASHBOARD_TOKEN"):
        print("Auth: DASHBOARD_TOKEN is set — clients must pass ?token=... or X-Dashboard-Token header.")
    else:
        print("Auth: DASHBOARD_TOKEN is NOT set — anyone with the URL can view it.")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Append-only JSONL journal with rotation.

Rotates the active file when EITHER:
    * the UTC date changes (daily rotation), OR
    * the file size exceeds `max_bytes`.

Rotated files are renamed to ``<stem>.YYYY-MM-DD[.N].jsonl`` and kept
up to ``backup_count``. Older files beyond that are deleted so a
multi-week run can never fill the disk.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class JsonlJournal:
    def __init__(
        self,
        path: Path,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB per file
        backup_count: int = 14,             # keep ~2 weeks of dailies
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._lock = threading.Lock()
        self._current_date = self._utc_date()

    @staticmethod
    def _utc_date() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ----- public API -----
    def write(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._maybe_rotate()
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")

    # ----- rotation -----
    def _maybe_rotate(self) -> None:
        today = self._utc_date()
        rotated = False

        if today != self._current_date and self.path.exists() and self.path.stat().st_size > 0:
            self._rotate_to(self._current_date)
            self._current_date = today
            rotated = True

        if (
            not rotated
            and self.path.exists()
            and self.path.stat().st_size >= self.max_bytes
        ):
            self._rotate_to(today)

        self._cleanup_old()

    def _rotate_to(self, date_label: str) -> None:
        # Find a free suffix like .2026-04-08.jsonl, .2026-04-08.1.jsonl, ...
        n = 0
        while True:
            suffix = f".{date_label}.jsonl" if n == 0 else f".{date_label}.{n}.jsonl"
            candidate = self.path.with_suffix("").with_name(self.path.stem + suffix)
            if not candidate.exists():
                break
            n += 1
        os.replace(self.path, candidate)

    def _cleanup_old(self) -> None:
        if self.backup_count <= 0:
            return
        rotated: List[Path] = sorted(
            p for p in self.path.parent.glob(self.path.stem + ".*.jsonl")
        )
        excess = len(rotated) - self.backup_count
        for old in rotated[: max(0, excess)]:
            try:
                old.unlink()
            except OSError:
                pass

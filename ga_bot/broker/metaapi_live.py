"""Live broker via MetaApi cloud SDK (Ubuntu deployment).

This file is intentionally a thin shim. It is **not imported by default**
so the GA training pipeline has zero dependency on `metaapi-cloud-sdk`.
Install it only when you actually go live:

    pip install metaapi-cloud-sdk

and provide credentials through environment variables:

    METAAPI_TOKEN          (your MetaApi auth token)
    METAAPI_ACCOUNT_ID     (the connected MT5 account id)

Then in `run_live.py` you swap `PaperBroker` for `MetaApiLiveBroker`.
The trader loop is unchanged.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from ..config import CONFIG
from .base import Broker, Order, Position, Side


class MetaApiLiveBroker(Broker):
    def __init__(self, token: Optional[str] = None, account_id: Optional[str] = None):
        self.token = token or os.environ.get("METAAPI_TOKEN")
        self.account_id = account_id or os.environ.get("METAAPI_ACCOUNT_ID")
        if not self.token or not self.account_id:
            raise RuntimeError(
                "MetaApi credentials missing. Set METAAPI_TOKEN and METAAPI_ACCOUNT_ID."
            )
        self._api = None
        self._account = None
        self._connection = None

    # ----- lifecycle -----
    def connect(self) -> None:
        try:
            from metaapi_cloud_sdk import MetaApi  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "metaapi-cloud-sdk is not installed. Run `pip install metaapi-cloud-sdk`."
            ) from exc

        import asyncio

        async def _connect():
            self._api = MetaApi(self.token)
            self._account = await self._api.metatrader_account_api.get_account(self.account_id)
            if self._account.state != "DEPLOYED":
                await self._account.deploy()
            await self._account.wait_connected()
            self._connection = self._account.get_streaming_connection()
            await self._connection.connect()
            await self._connection.wait_synchronized()

        asyncio.get_event_loop().run_until_complete(_connect())

    def disconnect(self) -> None:
        import asyncio
        if self._connection is not None:
            asyncio.get_event_loop().run_until_complete(self._connection.close())

    # ----- account -----
    def get_balance(self) -> float:
        info = self._connection.terminal_state.account_information
        return float(info["balance"])

    def get_equity(self) -> float:
        info = self._connection.terminal_state.account_information
        return float(info["equity"])

    def get_open_positions(self) -> List[Position]:
        out: List[Position] = []
        for p in self._connection.terminal_state.positions:
            side = Side.BUY if p["type"] == "POSITION_TYPE_BUY" else Side.SELL
            out.append(
                Position(
                    id=str(p["id"]),
                    symbol=p["symbol"],
                    side=side,
                    units=float(p["volume"]) * CONFIG.instrument.contract_size,
                    entry_price=float(p["openPrice"]),
                    sl=float(p.get("stopLoss", 0.0) or 0.0),
                    tp=float(p.get("takeProfit", 0.0) or 0.0),
                    opened_at=datetime.utcnow(),
                )
            )
        return out

    # ----- data -----
    def get_latest_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame:
        import asyncio

        tf_map = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h"}
        tf = tf_map.get(timeframe_minutes, "5m")
        end = datetime.utcnow()
        start = end - timedelta(minutes=timeframe_minutes * count * 2)

        async def _fetch():
            return await self._account.get_historical_candles(symbol, tf, start, count)

        candles = asyncio.get_event_loop().run_until_complete(_fetch())
        rows = [
            {
                "timestamp": pd.to_datetime(c["time"]),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c.get("tickVolume", 0)),
            }
            for c in candles
        ]
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df

    # ----- orders -----
    def place_order(self, order: Order) -> Optional[Position]:
        import asyncio

        # MetaApi takes lots, not units; 1 lot = contract_size oz of gold.
        volume_lots = order.units / CONFIG.instrument.contract_size

        async def _place():
            if order.side == Side.BUY:
                return await self._connection.create_market_buy_order(
                    order.symbol, volume_lots, order.sl, order.tp,
                    {"comment": order.comment or "ga_bot"},
                )
            return await self._connection.create_market_sell_order(
                order.symbol, volume_lots, order.sl, order.tp,
                {"comment": order.comment or "ga_bot"},
            )

        result = asyncio.get_event_loop().run_until_complete(_place())
        if not result or "positionId" not in result:
            return None
        return Position(
            id=str(result["positionId"]),
            symbol=order.symbol,
            side=order.side,
            units=order.units,
            entry_price=0.0,  # filled in by next sync
            sl=order.sl,
            tp=order.tp,
            opened_at=datetime.utcnow(),
        )

    def close_position(self, position_id: str) -> None:
        import asyncio

        async def _close():
            await self._connection.close_position(position_id)

        asyncio.get_event_loop().run_until_complete(_close())

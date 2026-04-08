"""Abstract broker interface.

The trader loop only ever talks to this interface. The same chromosome
can run against `PaperBroker` (phase 1) or a real-money broker (phase 2)
without any code changes outside this directory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import List, Optional

import pandas as pd


class Side(IntEnum):
    BUY = 1
    SELL = -1


@dataclass
class Order:
    symbol: str
    side: Side
    units: float
    sl: float
    tp: float
    comment: str = ""


@dataclass
class Position:
    id: str
    symbol: str
    side: Side
    units: float
    entry_price: float
    sl: float
    tp: float
    opened_at: datetime


class Broker(ABC):
    """Minimal interface every broker (paper or live) must implement."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_balance(self) -> float: ...

    @abstractmethod
    def get_equity(self) -> float: ...

    @abstractmethod
    def get_open_positions(self) -> List[Position]: ...

    @abstractmethod
    def get_latest_bars(self, symbol: str, timeframe_minutes: int, count: int) -> pd.DataFrame: ...

    @abstractmethod
    def place_order(self, order: Order) -> Optional[Position]: ...

    @abstractmethod
    def close_position(self, position_id: str) -> None: ...

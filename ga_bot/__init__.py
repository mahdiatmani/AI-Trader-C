"""Genetic Algorithm trading bot for XAU/USD (gold) on the 5-minute timeframe.

Phase 1: paper trading. The broker layer is abstracted so the same trained
chromosome can be promoted to live trading via MetaApi without touching the
strategy or backtester.
"""

__version__ = "0.1.0"

"""Seeded synthetic market data, so audits and experiments are reproducible offline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_ohlcv(n: int = 600, seed: int = 0, start: str = "2015-01-01") -> pd.DataFrame:
    """Random-walk daily bars with columns open, high, low, close, volume.

    Close follows a geometric random walk with 1% daily volatility and no
    drift.  Open gaps slightly from the previous close, and high and low
    bracket both.  The bars satisfy ``low <= min(open, close)`` and
    ``high >= max(open, close)``.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    previous = np.concatenate(([100.0], close[:-1]))
    open_ = previous * np.exp(rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.004, n)))
    volume = np.round(rng.lognormal(mean=13.0, sigma=0.4, size=n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.bdate_range(start, periods=n, name="date"),
    )

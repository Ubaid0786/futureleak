from __future__ import annotations

import pandas as pd
import pytest

from futureleak import synthetic_ohlcv


@pytest.fixture(scope="session")
def bars() -> pd.DataFrame:
    """400 synthetic daily bars on a business-day DatetimeIndex."""
    return synthetic_ohlcv(400, seed=7)


@pytest.fixture(scope="session")
def close(bars: pd.DataFrame) -> pd.Series:
    return bars["close"]

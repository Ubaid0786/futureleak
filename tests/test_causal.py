"""Functions that only use rows up to ``t`` must pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futureleak import assert_causal, audit


def _rsi(d: pd.DataFrame, window: int = 14) -> pd.Series:
    change = d["close"].diff()
    gain = change.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def _expanding_zscore(d: pd.DataFrame) -> pd.Series:
    x = d["close"]
    return (x - x.expanding().mean()) / x.expanding().std()


def _rolling_zscore(d: pd.DataFrame) -> pd.Series:
    x = d["close"]
    return (x - x.rolling(30).mean()) / x.rolling(30).std()


CAUSAL_FUNCTIONS = {
    "rolling_mean": lambda d: d["close"].rolling(20).mean(),
    "rolling_std": lambda d: d["close"].rolling(20).std(),
    "rolling_max": lambda d: d["high"].rolling(15).max(),
    "rolling_min_periods": lambda d: d["close"].rolling(20, min_periods=1).mean(),
    "rolling_median": lambda d: d["close"].rolling(11).median(),
    "rolling_rank": lambda d: d["close"].rolling(25).rank(pct=True),
    "rolling_corr": lambda d: d["close"].rolling(30).corr(d["volume"]),
    "rolling_apply": lambda d: d["close"].rolling(10).apply(lambda w: w[-1] - w[0], raw=True),
    "ewm_mean": lambda d: d["close"].ewm(span=20).mean(),
    "ewm_adjust_false": lambda d: d["close"].ewm(alpha=0.1, adjust=False).mean(),
    "ewm_std": lambda d: d["close"].ewm(halflife=10).std(),
    "expanding_mean": lambda d: d["close"].expanding().mean(),
    "expanding_std": lambda d: d["close"].expanding().std(),
    "expanding_max": lambda d: d["close"].expanding().max(),
    "expanding_zscore": _expanding_zscore,
    "rolling_zscore": _rolling_zscore,
    "lag_1": lambda d: d["close"].shift(1),
    "lag_20": lambda d: d["close"].shift(20),
    "diff": lambda d: d["close"].diff(3),
    "pct_change": lambda d: d["close"].pct_change(5),
    "log_return": lambda d: np.log(d["close"]).diff(),
    "cumsum": lambda d: d["volume"].cumsum(),
    "cumprod": lambda d: (1 + d["close"].pct_change().fillna(0)).cumprod(),
    "cummax": lambda d: d["close"].cummax(),
    "drawdown": lambda d: d["close"] / d["close"].cummax() - 1,
    "ffill": lambda d: d["close"].where(d["close"] > d["open"]).ffill(),
    "rsi": _rsi,
    "identity": lambda d: d["close"],
    "comparison": lambda d: d["close"] > d["close"].shift(1),
    "lagged_label": lambda d: d["close"].pct_change(5).shift(-5).shift(5),
    "numpy_cumsum": lambda d: np.cumsum(d["close"].to_numpy()),
}


@pytest.mark.parametrize("name", sorted(CAUSAL_FUNCTIONS))
def test_causal_function_passes(bars, name):
    report = audit(CAUSAL_FUNCTIONS[name], bars)
    assert report.verdict == "causal", report.summary()
    assert not report.leaks
    for col in report.columns:
        assert col.kind is None
        assert col.horizon is None
        assert col.leak_fraction == 0.0
        assert col.truncation_leak_fraction == 0.0
        assert col.perturbation_leak_fraction == 0.0


def test_causal_frame_of_features(bars):
    def features(d):
        out = pd.DataFrame(index=d.index)
        out["ret_1"] = d["close"].pct_change()
        out["ret_5"] = d["close"].pct_change(5)
        out["vol_20"] = out["ret_1"].rolling(20).std()
        out["range"] = (d["high"] - d["low"]) / d["close"]
        out["volume_z"] = (d["volume"] - d["volume"].rolling(60).mean()) / d["volume"].rolling(
            60
        ).std()
        out["dow"] = d.index.day_of_week
        return out

    report = assert_causal(features, bars)
    assert len(report.columns) == 6
    assert report.verdict == "causal"


def test_assert_causal_returns_the_report(bars):
    report = assert_causal(lambda d: d["close"].rolling(5).mean(), bars, n_cuts=7, seed=3)
    assert report.seed == 3
    assert len(report.cuts) == 7


@pytest.mark.parametrize("seed", range(5))
def test_no_false_positive_across_seeds(bars, seed):
    report = audit(_rsi, bars, seed=seed, n_cuts=40)
    assert report.verdict == "causal"


def test_series_input(close):
    report = audit(lambda x: x.ewm(span=12).mean() - x.ewm(span=26).mean(), close)
    assert report.verdict == "causal"
    assert report.columns[0].name == "close"

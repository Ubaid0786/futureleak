from __future__ import annotations

import numpy as np
import pandas as pd

from futureleak import synthetic_ohlcv
from futureleak.perturb import perturb


def test_every_numeric_value_changes_and_structure_is_kept():
    data = synthetic_ohlcv(300, seed=1)
    out = perturb(data, np.random.default_rng(0))
    assert out.index.equals(data.index)
    assert list(out.columns) == list(data.columns)
    assert (out.dtypes == data.dtypes).all()
    assert not (out.to_numpy() == data.to_numpy()).any()
    # Positive columns stay positive, so log and ratio features remain defined.
    assert (out > 0).all().all()


def test_input_is_not_modified():
    data = synthetic_ohlcv(50, seed=1)
    before = data.copy(deep=True)
    perturb(data, np.random.default_rng(0))
    pd.testing.assert_frame_equal(data, before)


def test_missing_values_stay_missing():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    out = perturb(s, np.random.default_rng(0))
    assert out.isna().tolist() == [False, True, False, True, False]


def test_columns_with_non_positive_values_get_additive_noise():
    s = pd.Series(np.linspace(-1.0, 1.0, 101))
    out = perturb(s, np.random.default_rng(0))
    assert not (out == s).any()
    assert np.isfinite(out).all()
    zeros = perturb(pd.Series(np.zeros(20)), np.random.default_rng(0))
    assert (zeros != 0).all()


def test_integer_boolean_and_object_columns():
    n = 200
    data = pd.DataFrame(
        {
            "i": np.arange(n, dtype="int64"),
            "u": np.arange(n, dtype="uint8"),
            "b": np.arange(n) % 2 == 0,
            "s": [f"k{i}" for i in range(n)],
            "t": pd.date_range("2024-01-01", periods=n, freq="D"),
        }
    )
    out = perturb(data, np.random.default_rng(0))
    assert (out.dtypes == data.dtypes).all()
    assert (out["i"] != data["i"]).all()
    assert (out["u"] != data["u"]).all()
    assert (out["b"] != data["b"]).any()
    assert sorted(out["s"]) == sorted(data["s"]) and (out["s"] != data["s"]).any()
    assert (out["t"] != data["t"]).any()


def test_same_generator_state_gives_the_same_result():
    data = synthetic_ohlcv(40, seed=2)
    a = perturb(data, np.random.default_rng(5))
    b = perturb(data, np.random.default_rng(5))
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_bars_are_consistent_and_seeded():
    bars = synthetic_ohlcv(250, seed=9)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
    assert (bars["volume"] > 0).all()
    assert bars.index.is_monotonic_increasing and bars.index.is_unique
    pd.testing.assert_frame_equal(bars, synthetic_ohlcv(250, seed=9))
    assert not bars.equals(synthetic_ohlcv(250, seed=10))

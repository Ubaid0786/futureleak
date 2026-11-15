"""Functions that are known to use future rows must be flagged, with the right horizon."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futureleak import LeakageError, assert_causal, audit


def test_shift_minus_one(bars):
    report = audit(lambda d: d["close"].shift(-1), bars)
    col = report["close"]
    assert report.leaks and report.verdict == "leak"
    assert col.kind == "fixed_horizon"
    assert col.horizon == 1
    assert col.earliest_offset == 0
    assert col.leak_fraction == 1.0
    assert not col.horizon_to_end
    assert not col.truncation_sensitive


@pytest.mark.parametrize("k", [1, 2, 5, 13, 40])
def test_negative_shift_horizon_equals_shift(bars, k):
    col = audit(lambda d: d["close"].shift(-k), bars)["close"]
    assert col.kind == "fixed_horizon"
    assert col.horizon == k
    # The k rows ending at the cut all read past it.
    assert col.earliest_offset == -(k - 1)
    assert col.leak_fraction == 1.0


@pytest.mark.parametrize("window", [3, 5, 7, 21])
def test_centred_rolling_window(bars, window):
    col = audit(lambda d: d["close"].rolling(window, center=True).mean(), bars)["close"]
    assert col.verdict == "leak"
    assert col.kind == "fixed_horizon"
    assert col.horizon == window // 2
    assert col.earliest_offset == -(window // 2 - 1)


def test_global_zscore(bars):
    def zscore(d):
        return (d["close"] - d["close"].mean()) / d["close"].std()

    col = audit(zscore, bars)["close"]
    assert col.verdict == "leak"
    assert col.kind == "full_sample"
    assert col.horizon_to_end
    assert col.truncation_sensitive
    assert col.leak_fraction == 1.0
    assert col.truncation_leak_fraction == 1.0
    assert col.perturbation_leak_fraction == 1.0
    # Every row from the first one is contaminated.
    assert col.earliest_offset is not None and col.earliest_offset < -90


def test_global_min_max_scaling(bars):
    def scale(d):
        x = d["close"]
        return (x - x.min()) / (x.max() - x.min())

    col = audit(scale, bars)["close"]
    assert col.verdict == "leak"
    assert col.truncation_sensitive
    assert col.leak_fraction > 0.5


def test_full_sample_rank(bars):
    col = audit(lambda d: d["close"].rank(pct=True), bars)["close"]
    assert col.verdict == "leak"
    assert col.truncation_sensitive


def test_backfill():
    values = np.arange(200, dtype=float) + 1.0
    # Gaps of three rows: positions 10-12, 20-22, ...
    for start in range(10, 200, 10):
        values[start : start + 3] = np.nan
    s = pd.Series(values, name="x")

    report = audit(lambda x: x.bfill(), s, n_cuts=1000)
    col = report["x"]
    assert col.verdict == "leak"
    assert col.kind == "variable_horizon"
    assert col.truncation_sensitive
    # A gap row is filled from the first valid row after the gap.
    assert col.horizon == 3
    assert col.horizon_min == 1
    assert 0.2 < col.leak_fraction < 0.4


def test_interpolation_uses_next_valid_value():
    values = np.sin(np.arange(300) / 7.0) + 2.0
    values[5::9] = np.nan
    s = pd.Series(values, name="x")
    col = audit(lambda x: x.interpolate(), s, n_cuts=1000)["x"]
    assert col.verdict == "leak"
    assert col.horizon == 1


@pytest.mark.parametrize("h", [1, 5, 20])
def test_target_leakage_through_future_returns(bars, h):
    def future_return(d):
        return d["close"].shift(-h) / d["close"] - 1.0

    col = audit(future_return, bars)["close"]
    assert col.verdict == "leak"
    assert col.kind == "fixed_horizon"
    assert col.horizon == h


def test_only_the_leaking_column_is_flagged(bars):
    def features(d):
        return pd.DataFrame(
            {
                "momentum": d["close"].pct_change(10),
                "volatility": d["close"].pct_change().rolling(20).std(),
                "label": d["close"].pct_change(5).shift(-5),
            }
        )

    report = audit(features, bars)
    assert [c.name for c in report.leaking_columns] == ["label"]
    assert report["momentum"].verdict == "causal"
    assert report["volatility"].verdict == "causal"
    assert report["label"].horizon == 5


def test_weekly_aggregate_broadcast_to_days(bars):
    def weekly_mean(d):
        return d["close"].groupby(d.index.to_period("W")).transform("mean")

    col = audit(weekly_mean, bars, n_cuts=60)["close"]
    assert col.verdict == "leak"
    # Monday's value includes the four later days of the same week.
    assert col.horizon == 4


def test_length_dependent_output(bars):
    def position(d):
        return pd.Series(np.arange(len(d)) / len(d), index=d.index, name="pos")

    col = audit(position, bars)["pos"]
    assert col.verdict == "leak"
    assert col.kind == "truncation_only"
    assert col.horizon is None
    assert col.truncation_leak_fraction == 1.0
    assert col.perturbation_leak_fraction == 0.0


def test_start_of_series_filled_with_full_sample_mean(bars):
    def momentum(d):
        x = d["close"]
        return x - x.shift(10, fill_value=x.mean())

    report = audit(momentum, bars)
    col = report["close"]
    assert col.verdict == "leak"
    assert col.kind == "full_sample"
    assert col.last_affected_row == 9
    assert any("only rows up to position 9" in note for note in col.notes)


def test_numpy_output(bars):
    report = audit(lambda d: np.roll(d["close"].to_numpy(), -2), bars)
    assert report["value"].verdict == "leak"
    assert report["value"].horizon == 2


def test_boolean_output(bars):
    col = audit(lambda d: d["close"].shift(-1) > d["close"], bars, n_cuts=60)["close"]
    assert col.verdict == "leak"
    assert col.horizon == 1


def test_string_output(bars):
    def direction(d):
        up = d["close"].shift(-1) > d["close"]
        return pd.Series(np.where(up, "up", "down"), index=d.index, name="direction")

    col = audit(direction, bars, n_cuts=60)["direction"]
    assert col.verdict == "leak"
    assert col.horizon == 1


@pytest.mark.parametrize("h", [1, 3, 8])
def test_lagging_by_the_reported_horizon_repairs_a_fixed_horizon_leak(bars, h):
    def leaky(d):
        return d["close"].rolling(2 * h + 1, center=True).mean()

    horizon = audit(leaky, bars)["close"].horizon
    assert horizon == h
    assert audit(lambda d: leaky(d).shift(horizon), bars).verdict == "causal"
    assert audit(lambda d: leaky(d).shift(horizon - 1), bars).verdict == "leak"


def test_assert_causal_raises_with_report(bars):
    with pytest.raises(LeakageError) as info:
        assert_causal(lambda d: d["close"].shift(-3), bars)
    assert isinstance(info.value, AssertionError)
    assert info.value.report["close"].horizon == 3
    assert "look-ahead leakage" in str(info.value)


@pytest.mark.parametrize("seed", range(5))
def test_detection_does_not_depend_on_the_seed(bars, seed):
    col = audit(lambda d: d["close"].rolling(9, center=True).mean(), bars, seed=seed)["close"]
    assert col.verdict == "leak"
    assert col.horizon == 4

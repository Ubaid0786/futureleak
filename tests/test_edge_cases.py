"""Awkward functions and inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from futureleak import AlignmentError, AuditError, LeakageError, assert_causal, audit

# --- shorter outputs ------------------------------------------------------


def test_causal_output_with_warmup_rows_dropped(bars):
    report = audit(lambda d: d["close"].rolling(30).mean().dropna(), bars)
    assert report.verdict == "causal"


def test_leaky_output_with_trailing_rows_dropped(bars):
    col = audit(lambda d: d["close"].shift(-2).dropna(), bars)["close"]
    assert col.verdict == "leak"
    assert col.horizon == 2


def test_filtered_rows(bars):
    # Only up-days are returned; the selection itself is causal.
    report = audit(lambda d: d.loc[d["close"] > d["open"], "close"].rolling(3).mean(), bars)
    assert report.verdict == "causal"


def test_unlabelled_output_of_the_wrong_length_is_rejected(bars):
    with pytest.raises(AlignmentError, match="rows cannot be matched"):
        audit(lambda d: d["close"].to_numpy()[5:], bars)


def test_same_length_output_with_a_reset_index_is_matched_by_position(bars):
    report = audit(lambda d: d["close"].rolling(5).mean().reset_index(drop=True), bars)
    assert report.verdict == "causal"
    leaky = audit(lambda d: d["close"].shift(-1).reset_index(drop=True), bars)
    assert leaky.verdict == "leak"


# --- warm-up --------------------------------------------------------------


def test_cuts_inside_the_warmup_period(bars):
    report = audit(lambda d: d["close"].rolling(150).mean(), bars, min_cut=1, n_cuts=50)
    assert report.verdict == "causal"
    assert min(report.cuts) < 150


def test_leak_is_found_when_most_cuts_fall_in_the_warmup(bars):
    col = audit(lambda d: d["close"].rolling(301, center=True).mean(), bars, min_cut=1, n_cuts=40)[
        "close"
    ]
    assert col.verdict == "leak"
    assert 0 < col.leak_fraction < 1


def test_output_that_is_entirely_missing_is_inconclusive(bars):
    report = audit(lambda d: d["close"].rolling(1000).mean(), bars)
    assert report.verdict == "inconclusive"
    assert "entirely missing" in report["close"].notes[0]
    with pytest.raises(LeakageError, match="inconclusive"):
        assert_causal(lambda d: d["close"].rolling(1000).mean(), bars)
    assert_causal(lambda d: d["close"].rolling(1000).mean(), bars, allow_inconclusive=True)


def test_constant_output_is_reported_as_low_power(bars):
    report = audit(lambda d: pd.Series(1.0, index=d.index, name="one"), bars)
    assert report.verdict == "causal"
    assert "little power" in report["one"].notes[0]


# --- non-determinism ------------------------------------------------------


def test_random_function_is_not_reported_as_a_leak(bars):
    rng = np.random.default_rng(1)

    def noisy(d):
        return d["close"] + rng.normal(size=len(d))

    report = audit(noisy, bars)
    assert report.verdict == "non_deterministic"
    assert not report.leaks
    assert report.non_deterministic
    assert report["close"].leak_fraction == 0.0
    assert report.notes


def test_non_determinism_is_tracked_per_column(bars):
    rng = np.random.default_rng(2)

    def mixed(d):
        return pd.DataFrame(
            {
                "noise": rng.normal(size=len(d)),
                "clean": d["close"].rolling(5).mean(),
                "leaky": d["close"].shift(-1),
            },
            index=d.index,
        )

    report = audit(mixed, bars)
    assert report["noise"].verdict == "non_deterministic"
    assert report["clean"].verdict == "causal"
    assert report["leaky"].verdict == "leak"
    assert report.verdict == "leak"


def test_assert_causal_on_a_random_function(bars):
    rng = np.random.default_rng(3)

    def noisy(d):
        return d["close"] * rng.uniform(0.9, 1.1, len(d))

    with pytest.raises(LeakageError, match="not deterministic"):
        assert_causal(noisy, bars)
    report = assert_causal(noisy, bars, allow_non_deterministic=True)
    assert report.non_deterministic


def test_seeded_function_is_deterministic(bars):
    def seeded(d):
        noise = np.random.default_rng(0).normal(size=len(d))
        return d["close"].rolling(5).mean() + noise

    assert audit(seeded, bars).verdict == "causal"


# --- functions that raise -------------------------------------------------


def test_function_that_raises_on_short_input_is_still_audited(bars):
    def needs_history(d):
        if len(d) < 250:
            raise ValueError("need 250 rows")
        return d["close"].rolling(20).mean()

    report = audit(needs_history, bars)
    assert report.verdict == "causal"
    assert 0 < report.truncation_errors < len(report.cuts)
    assert report.perturbation_errors == 0
    assert "raised on the truncated input" in report.to_markdown()


def test_leak_is_found_when_every_truncated_call_raises(bars):
    def needs_everything(d):
        if len(d) < len(bars):
            raise ValueError("need the full sample")
        return d["close"].shift(-4)

    report = audit(needs_everything, bars)
    assert report.truncation_errors == len(report.cuts)
    col = report["close"]
    assert col.verdict == "leak"
    assert col.horizon == 4
    assert col.truncation_leak_fraction is None


def test_leak_is_reported_when_the_horizon_search_fails(bars):
    original = bars["close"].to_numpy().copy()

    def fragile(d):
        # Raises when only a few trailing rows differ from the original, which is
        # what the horizon search feeds it.
        if len(d) == len(original) and 0 < (d["close"].to_numpy() != original).sum() < 3:
            raise ValueError("unexpected input")
        return d["close"].shift(-4)

    report = audit(fragile, bars, n_cuts=5, seed=0)
    assert max(report.cuts) < len(bars) - 3
    col = report["close"]
    assert col.verdict == "leak"
    assert col.kind == "variable_horizon"
    assert col.horizon is None
    assert any("could not be measured" in note for note in col.notes)


def test_function_that_raises_on_the_full_input(bars):
    def broken(d):
        raise RuntimeError("boom")

    with pytest.raises(AuditError, match="RuntimeError.*boom"):
        audit(broken, bars)


def test_function_that_returns_none(bars):
    with pytest.raises(AuditError, match="returned None"):
        audit(lambda d: None, bars)


def test_function_that_returns_an_unsupported_type(bars):
    with pytest.raises(AuditError, match="type float"):
        audit(lambda d: 1.5, bars)


# --- output and input types -----------------------------------------------


def test_unnamed_series_gets_a_default_column_name(bars):
    report = audit(lambda d: pd.Series(d["close"].to_numpy(), index=d.index), bars)
    assert [c.name for c in report.columns] == ["value"]


def test_tuple_output(bars):
    def bands(d):
        mid = d["close"].rolling(20).mean()
        width = d["close"].rolling(20, center=True).std()
        return mid, width

    report = audit(bands, bars)
    assert [c.name for c in report.columns] == ["out0", "out1"]
    assert report["out0"].verdict == "causal"
    assert report["out1"].verdict == "leak"


def test_mapping_output(bars):
    report = audit(
        lambda d: {"lag": d["close"].shift(1), "lead": d["close"].shift(-1).to_numpy()}, bars
    )
    assert report["lag"].verdict == "causal"
    assert report["lead"].verdict == "leak"


def test_two_dimensional_array_output(bars):
    def stacked(d):
        x = d["close"].to_numpy()
        return np.column_stack([np.cumsum(x), np.roll(x, -1)])

    report = audit(stacked, bars)
    assert report["0"].verdict == "causal"
    assert report["1"].verdict == "leak"


def test_duplicate_column_names_are_kept_apart(bars):
    def duplicated(d):
        out = pd.concat([d["close"].shift(1), d["close"].shift(-1)], axis=1)
        out.columns = ["x", "x"]
        return out

    report = audit(duplicated, bars)
    assert [c.name for c in report.columns] == ["x", "x#1"]
    assert report["x"].verdict == "causal"
    assert report["x#1"].verdict == "leak"


def test_integer_index(bars):
    data = bars.reset_index(drop=True)
    assert audit(lambda d: d["close"].rolling(5).mean(), data).verdict == "causal"
    col = audit(lambda d: d["close"].shift(-3), data)["close"]
    assert col.verdict == "leak" and col.horizon == 3


def test_integer_index_that_does_not_start_at_zero(bars):
    data = bars.set_axis(np.arange(1000, 1000 + len(bars)), axis=0)
    assert audit(lambda d: d["close"].rolling(5).mean().dropna(), data).verdict == "causal"
    assert audit(lambda d: d["close"].shift(-1).dropna(), data).verdict == "leak"


def test_duplicate_index_labels_fall_back_to_position(bars):
    index = bars.index.repeat(2)[: len(bars)]
    data = bars.set_axis(index, axis=0)
    assert audit(lambda d: d["close"].rolling(5).mean(), data).verdict == "causal"
    assert audit(lambda d: d["close"].shift(-1), data).verdict == "leak"


def test_index_frequency_survives_truncation_and_perturbation(bars):
    assert bars.index.freq is not None

    def needs_freq(d):
        if d.index.freq is None:
            raise ValueError("index lost its frequency")
        return d["close"].shift(1, freq=d.index.freq).reindex(d.index)

    report = audit(needs_freq, bars)
    assert report.truncation_errors == 0
    assert report.perturbation_errors == 0
    assert report.verdict == "causal"


def test_mixed_input_dtypes():
    n = 300
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "price": 50 + rng.normal(size=n).cumsum(),
            "count": rng.integers(0, 100, n),
            "flag": rng.random(n) < 0.5,
            "venue": rng.choice(["A", "B", "C"], n),
            "when": pd.date_range("2024-01-01", periods=n, freq="h"),
        }
    )

    def causal(d):
        return pd.DataFrame(
            {
                "count_sum": d["count"].rolling(5).sum(),
                "flag_rate": d["flag"].astype(float).expanding().mean(),
                "venue_changed": d["venue"] != d["venue"].shift(1),
                "hour": d["when"].dt.hour,
            }
        )

    def leaky(d):
        return pd.DataFrame(
            {
                "next_count": d["count"].shift(-1),
                "next_flag": d["flag"].shift(-1),
                "next_venue": d["venue"].shift(-1),
                "next_when": d["when"].shift(-1),
            }
        )

    assert audit(causal, data, n_cuts=40).verdict == "causal"
    report = audit(leaky, data, n_cuts=40)
    for name in ("next_count", "next_flag", "next_venue", "next_when"):
        assert report[name].verdict == "leak", name
        assert report[name].horizon == 1, name


def test_nullable_dtypes():
    n = 200
    rng = np.random.default_rng(1)
    data = pd.DataFrame(
        {
            "x": pd.array(rng.normal(size=n), dtype="Float64"),
            "k": pd.array(rng.integers(1, 9, n), dtype="Int64"),
        }
    )
    data.loc[10, "x"] = pd.NA
    assert audit(lambda d: d["x"].rolling(4, min_periods=1).mean(), data).verdict == "causal"
    assert audit(lambda d: d["k"].shift(-1), data).verdict == "leak"


def test_function_that_mutates_its_input(bars):
    before = bars.copy(deep=True)

    def mutating(d):
        d["ma"] = d["close"].rolling(5).mean()
        d["close"] = 0.0
        return d["ma"]

    assert audit(mutating, bars).verdict == "causal"
    pd.testing.assert_frame_equal(bars, before)


# --- tolerance ------------------------------------------------------------


def test_differences_below_the_tolerance_are_ignored(bars):
    def nearly(d):
        return d["close"] + 1e-13 * d["close"].shift(-1).fillna(0.0)

    assert audit(nearly, bars).verdict == "causal"
    strict = audit(nearly, bars, rtol=0.0, atol=0.0)
    assert strict.verdict == "leak"
    assert strict["close"].max_abs_diff is not None and strict["close"].max_abs_diff < 1e-9


def test_infinite_values_compare_equal(bars):
    def with_inf(d):
        out = d["close"].rolling(5).mean()
        out.iloc[::7] = np.inf
        out.iloc[3::7] = -np.inf
        return out

    assert audit(with_inf, bars).verdict == "causal"


# --- arguments ------------------------------------------------------------


def test_cut_points(bars):
    report = audit(lambda d: d["close"], bars, n_cuts=15, seed=4)
    assert len(report.cuts) == 15
    assert list(report.cuts) == sorted(set(report.cuts))
    assert min(report.cuts) >= len(bars) // 4
    assert max(report.cuts) <= len(bars) - 2


def test_more_cuts_than_rows_uses_every_position():
    data = pd.Series(np.arange(12.0), name="x")
    report = audit(lambda x: x.shift(-1), data, n_cuts=500, min_cut=0)
    assert report.cuts == tuple(range(0, 11))
    assert report["x"].verdict == "leak"


def test_same_seed_gives_the_same_report(bars):
    func = lambda d: d["close"].rolling(7, center=True).mean()  # noqa: E731
    assert audit(func, bars, seed=11).to_dict() == audit(func, bars, seed=11).to_dict()
    assert audit(func, bars, seed=11).cuts != audit(func, bars, seed=12).cuts


def test_invalid_arguments(bars):
    with pytest.raises(TypeError, match="DataFrame or Series"):
        audit(lambda d: d, [1, 2, 3])
    with pytest.raises(ValueError, match="at least 4 rows"):
        audit(lambda d: d, bars.iloc[:3])
    with pytest.raises(ValueError, match="n_cuts"):
        audit(lambda d: d, bars, n_cuts=0)
    with pytest.raises(ValueError, match="min_cut"):
        audit(lambda d: d, bars, min_cut=len(bars))


def test_report_name(bars):
    assert audit(lambda d: d["close"], bars, name="my_feature").function == "my_feature"

    def named(d):
        return d["close"]

    assert audit(named, bars).function.endswith("named")

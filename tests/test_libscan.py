"""Library scanning.  Tests that need an indicator library are skipped when it is absent."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from futureleak import audit, synthetic_ohlcv
from futureleak.cli import main
from futureleak.libscan import (
    Indicator,
    ScanResult,
    _scan_one,
    _split_parameters,
    discover,
    horizon_text,
    library_version,
    resolve_library,
    results_to_markdown,
    results_to_text,
    scan_indicators,
    scan_library,
    summarise,
    ta_indicators,
)


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return synthetic_ohlcv(400, seed=3)


def _find(indicators: list[Indicator], name: str) -> Indicator:
    return next(i for i in indicators if i.name == name)


# --- helpers that need no library -----------------------------------------


def test_resolve_library_accepts_distribution_names():
    assert resolve_library("ta") == "ta"
    assert resolve_library("TA-Lib") == "talib"
    assert resolve_library("pandas-ta-classic") == "pandas_ta_classic"
    assert resolve_library("bta-lib") == "btalib"
    with pytest.raises(KeyError, match="unsupported library"):
        resolve_library("nope")


def test_split_parameters():
    def indicator(high, low, close, window=14, fillna=False, **kwargs):
        return None

    def needs_extra(close, benchmark, window=5):
        return None

    aliases = {name: name for name in ("open", "high", "low", "close", "volume")}
    mapping, missing = _split_parameters(inspect.signature(indicator), aliases)
    assert mapping == {"high": "high", "low": "low", "close": "close"} and missing is None
    mapping, missing = _split_parameters(inspect.signature(needs_extra), aliases)
    assert missing == "benchmark"


def test_scan_one_outcomes(data):
    def lead(df):
        return df["close"].shift(-2).rename("lead")

    def positional(df):
        # Label lookup with an integer: fails on a date index, works on a range index.
        if df["close"].loc[0] > 0:
            return df["close"].rolling(3).mean()

    def table(df):
        return pd.DataFrame({"bin": range(10)})

    def broken(df):
        raise ZeroDivisionError("division by zero")

    def run(func):
        return _scan_one(Indicator("lib", func.__name__, "call", func), "1.0", data, 8, 0)

    leak = run(lead)
    assert (leak.status, leak.verdict) == ("audited", "leak")
    assert leak.detail == "lead: fixed_horizon, horizon 2, 100% of cuts"
    assert horizon_text(leak) == "2"

    retried = run(positional)
    assert (retried.status, retried.verdict) == ("audited", "causal")
    assert retried.detail.startswith("audited on an integer index (on the date index: KeyError")

    skipped = run(table)
    assert skipped.status == "skipped"
    assert skipped.detail == "output is not one row per input row"

    failed = run(broken)
    assert failed.status == "error"
    assert failed.detail == "ZeroDivisionError: division by zero"
    assert failed.verdict is None and horizon_text(failed) == "-"

    results = [leak, retried, skipped, failed]
    counts = summarise(results)
    assert counts == {
        "found": 4,
        "audited": 2,
        "causal": 1,
        "leak": 1,
        "inconclusive": 0,
        "non_deterministic": 0,
        "error": 1,
        "skipped": 1,
    }
    text = results_to_text(results)
    assert text.splitlines()[0].startswith(
        "lib 1.0: 4 indicators found, 2 audited (1 causal, 1 leak"
    )
    markdown = results_to_markdown(results)
    assert markdown.count("\n") == 2 + 4
    assert "| lib | 1.0 | lead | leak | 2 |" in markdown
    payload = leak.to_dict()
    assert payload["columns"][0]["horizon"] == 2 and payload["status"] == "audited"


def test_results_to_text_when_empty():
    assert results_to_text([]) == "no indicators found"
    assert isinstance(ScanResult("lib", "1", "x", "skipped").to_dict(), dict)


# --- real libraries -------------------------------------------------------


def test_ta_scan(data):
    pytest.importorskip("ta")
    results = scan_library("ta", data, n_cuts=6)
    by_name = {r.indicator: r for r in results}
    assert len(results) > 30
    assert by_name["momentum.RSIIndicator"].verdict == "causal"
    assert by_name["volatility.BollingerBands"].verdict == "causal"
    assert all(r.library == "ta" and r.version for r in results)
    assert [r.indicator.lower() for r in results] == sorted(r.indicator.lower() for r in results)


def test_ta_fillna_option_is_passed_through(data):
    pytest.importorskip("ta")
    default, _ = ta_indicators()
    filled, skipped = ta_indicators(fillna=True)
    assert len(default) == len(filled)
    assert all("fillna=True" in i.call for i in filled)
    assert not any("fillna" in i.call for i in default)

    # With fillna=True the indicator returns its warm-up rows, which hold the
    # mean of the whole series (ta/trend.py, DPOIndicator._run).
    dpo = [i for i in filled if i.name == "trend.DPOIndicator"]
    results = scan_indicators(dpo, skipped, version=library_version("ta"), data=data, n_cuts=6)
    by_name = {r.indicator: r for r in results}
    assert by_name["trend.DPOIndicator"].verdict == "leak"
    assert "only rows up to position 10 are affected" in by_name["trend.DPOIndicator"].detail
    assert by_name["trend.SMAIndicator"].status == "skipped"
    assert audit(_find(default, "trend.DPOIndicator").func, data, n_cuts=6).verdict == "causal"


def test_finta_ichimoku_chikou_span_and_fractals(data):
    pytest.importorskip("finta")
    indicators, skipped = discover("finta")
    assert len(indicators) > 60
    assert any(s.name == "LWMA" for s in skipped)

    report = audit(_find(indicators, "ICHIMOKU").func, data, n_cuts=8)
    assert [c.name for c in report.leaking_columns] == ["CHIKOU"]
    assert report["CHIKOU"].kind == "fixed_horizon" and report["CHIKOU"].horizon == 26
    assert report["TENKAN"].verdict == "causal"

    fractal = audit(_find(indicators, "WILLIAMS_FRACTAL").func, data, n_cuts=40)
    assert fractal.verdict == "leak"
    assert max(c.horizon for c in fractal.leaking_columns) == 2

    assert audit(_find(indicators, "SMA").func, data, n_cuts=8).verdict == "causal"


def test_stockstats_families(data):
    pytest.importorskip("stockstats")
    indicators, skipped = discover("stockstats")
    names = {i.name for i in indicators}
    assert not skipped
    assert {"macd", "boll", "eribull"} <= names
    # The bare family keys are not columns and must not be listed separately.
    assert "eri" not in names and "wt" not in names
    report = audit(_find(indicators, "macd").func, data, n_cuts=6)
    assert [c.name for c in report.columns] == ["macd", "macds", "macdh"]
    assert report.verdict == "causal"


def test_pandas_ta_classic_documented_non_causal_indicators(data):
    pytest.importorskip("pandas_ta_classic")
    indicators, skipped = discover("pandas_ta_classic")
    assert any(s.name == "math.add" for s in skipped)

    dpo = audit(_find(indicators, "trend.dpo").func, data, n_cuts=8)
    assert dpo.verdict == "leak" and dpo.columns[0].horizon == 11

    ichimoku = audit(_find(indicators, "overlap.ichimoku").func, data, n_cuts=8)
    assert [c.name for c in ichimoku.leaking_columns] == ["ICS_26"]

    assert audit(_find(indicators, "momentum.rsi").func, data, n_cuts=8).verdict == "causal"


def test_talib_functions(data):
    pytest.importorskip("talib")
    indicators, skipped = discover("talib")
    assert any(s.name == "MAVP" for s in skipped)
    macd = audit(_find(indicators, "MACD").func, data, n_cuts=6)
    assert [c.name for c in macd.columns] == ["macd", "macdsignal", "macdhist"]
    assert macd.verdict == "causal"


@pytest.mark.filterwarnings("ignore")
def test_btalib_recovers_after_an_indicator_raises(data):
    pytest.importorskip("btalib")
    indicators, _ = discover("btalib")
    for name in ("ema", "stochastic", "sma"):
        try:
            _find(indicators, name).func(data.copy())
        except Exception:
            pass
    report = audit(_find(indicators, "sma").func, data, n_cuts=6)
    assert report.verdict == "causal"


def test_cli_scan_lib(capsys):
    pytest.importorskip("stockstats")
    code = main(
        ["scan-lib", "stockstats", "--n-cuts", "4", "--rows", "300", "--format", "markdown"]
    )
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert out.startswith("| library | version | indicator | verdict | horizon (rows) | notes |")
    assert "| stockstats |" in out

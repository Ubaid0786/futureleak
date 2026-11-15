from __future__ import annotations

import json

import pandas as pd
import pytest

from futureleak import ColumnReport, Report, audit


@pytest.fixture(scope="module")
def report(bars) -> Report:
    def features(d):
        return pd.DataFrame(
            {
                "sma": d["close"].rolling(10).mean(),
                "lead": d["close"].shift(-2),
                "z": (d["close"] - d["close"].mean()) / d["close"].std(),
            }
        )

    return audit(features, bars, name="features")


def test_properties(report):
    assert report.leaks
    assert report.verdict == "leak"
    assert [c.name for c in report.leaking_columns] == ["lead", "z"]
    assert isinstance(report["sma"], ColumnReport)
    assert report["lead"].horizon_text() == "2"
    assert report["z"].horizon_text().endswith("(to end of data)")
    assert report["sma"].horizon_text() == "-"
    with pytest.raises(KeyError):
        report["missing"]


def test_to_dict_and_json_round_trip(report):
    as_dict = report.to_dict()
    assert as_dict["function"] == "features"
    assert as_dict["verdict"] == "leak"
    assert as_dict["n_rows"] == 400
    assert len(as_dict["cuts"]) == 20
    assert [c["name"] for c in as_dict["columns"]] == ["sma", "lead", "z"]
    lead = as_dict["columns"][1]
    assert lead["kind"] == "fixed_horizon" and lead["horizon"] == 2
    assert json.loads(report.to_json()) == as_dict
    assert json.loads(report.to_json(indent=None)) == as_dict


def test_markdown(report):
    text = report.to_markdown()
    lines = text.splitlines()
    assert lines[0] == "### futureleak audit: `features`"
    assert "Verdict: **leak**" in text
    table = [line for line in lines if line.startswith("|")]
    assert len(table) == 2 + 3
    assert all(line.count("|") == table[0].count("|") for line in table)
    assert "| lead | leak | fixed_horizon | 100% | 2 | -1 |" in text


def test_markdown_escapes_pipes(bars):
    report = audit(lambda d: d["close"].rename("a|b"), bars)
    assert "a\\|b" in report.to_markdown()


def test_summary_and_str(report):
    text = str(report)
    assert text == report.summary()
    assert text.splitlines()[0] == "features: leak (400 rows, 20 cuts)"
    assert "lead: leak [fixed_horizon]" in text
    assert "sma: causal" in text


def test_verdict_precedence():
    def make(*verdicts: str) -> Report:
        return Report(
            function="f",
            n_rows=10,
            cuts=(5,),
            seed=0,
            rtol=1e-9,
            atol=1e-12,
            columns=tuple(ColumnReport(name=str(i), verdict=v) for i, v in enumerate(verdicts)),
        )

    assert make("causal", "causal").verdict == "causal"
    assert make("causal", "inconclusive").verdict == "inconclusive"
    assert make("inconclusive", "non_deterministic").verdict == "non_deterministic"
    assert make("non_deterministic", "leak", "causal").verdict == "leak"


def test_reports_are_immutable(report):
    with pytest.raises(AttributeError):
        report.seed = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        report["sma"].verdict = "leak"  # type: ignore[misc]

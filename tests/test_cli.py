from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from futureleak import __version__, synthetic_ohlcv
from futureleak.cli import load_callable, load_csv, main

MODULE = textwrap.dedent(
    """
    import numpy as np


    def causal(df):
        return df["close"].rolling(10).mean()


    def leaky(df):
        return df["close"].shift(-3)


    def series_feature(close):
        return close.ewm(span=10).mean()


    def all_missing(df):
        return df["close"] * np.nan


    def broken(df):
        raise RuntimeError("boom")


    class Features:
        @staticmethod
        def lead(df):
            return df["close"].shift(-1)


    NOT_CALLABLE = 3
    """
)


@pytest.fixture()
def project(tmp_path):
    module = tmp_path / "feats.py"
    module.write_text(MODULE, encoding="utf-8")
    csv = tmp_path / "bars.csv"
    synthetic_ohlcv(300, seed=5).to_csv(csv)
    return module, csv


def test_audit_causal_function_exits_zero(project, capsys):
    module, csv = project
    code = main(["audit", f"{module}:causal", "--data", str(csv), "--index-col", "date"])
    assert code == 0
    assert "causal" in capsys.readouterr().out


def test_audit_leaky_function_exits_one(project, capsys):
    module, csv = project
    code = main(["audit", f"{module}:leaky", "--data", str(csv), "--index-col", "0"])
    out = capsys.readouterr().out
    assert code == 1
    assert "fixed_horizon" in out and "horizon 3" in out


def test_json_output(project, capsys):
    module, csv = project
    code = main(
        ["audit", f"{module}:leaky", "--data", str(csv), "--format", "json", "--n-cuts", "8"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["verdict"] == "leak"
    assert payload["columns"][0]["horizon"] == 3
    assert len(payload["cuts"]) == 8


def test_markdown_output_to_file(project, tmp_path, capsys):
    module, csv = project
    target = tmp_path / "report.md"
    code = main(
        [
            "audit",
            f"{module}:causal",
            "--data",
            str(csv),
            "--format",
            "markdown",
            "--output",
            str(target),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    assert target.read_text(encoding="utf-8").startswith("### futureleak audit")


def test_column_option_passes_a_series(project):
    module, csv = project
    assert main(["audit", f"{module}:series_feature", "--data", str(csv), "--column", "close"]) == 0


def test_synthetic_data_when_no_csv_is_given(project):
    module, _ = project
    assert main(["audit", f"{module}:Features.lead", "--rows", "200"]) == 1


def test_unverifiable_function_exits_three(project):
    module, csv = project
    assert main(["audit", f"{module}:all_missing", "--data", str(csv)]) == 3


@pytest.mark.parametrize(
    "target",
    ["broken", "does_not_exist", "NOT_CALLABLE"],
)
def test_errors_exit_two(project, capsys, target):
    module, csv = project
    assert main(["audit", f"{module}:{target}", "--data", str(csv)]) == 2
    assert "futureleak: error:" in capsys.readouterr().err


def test_bad_inputs_exit_two(project, capsys):
    module, csv = project
    assert main(["audit", "no_colon", "--data", str(csv)]) == 2
    assert main(["audit", f"{module}:causal", "--data", "missing.csv"]) == 2
    assert main(["audit", f"{module}:causal", "--data", str(csv), "--column", "nope"]) == 2
    assert main(["audit", "surely_not_a_module_xyz:f"]) == 2
    assert main(["scan-lib", "not-a-library"]) == 2
    assert "unsupported library" in capsys.readouterr().err


def test_load_callable_from_an_importable_module():
    assert load_callable("futureleak:audit").__name__ == "audit"
    assert load_callable("futureleak.cli:main") is main


def test_load_csv_parses_a_date_index(project):
    _, csv = project
    frame = load_csv(str(csv), "date")
    assert str(frame.index.dtype).startswith("datetime64")
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    plain = load_csv(str(csv), None)
    assert "date" in plain.columns and len(plain) == 300


@pytest.mark.parametrize("module", ["futureleak", "futureleak.cli"])
def test_module_entry_point(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"futureleak {__version__}"

"""Audit every indicator of the installed technical-analysis libraries for look-ahead.

Each indicator is called with its default parameters on seeded synthetic daily
bars.  The indicators are found by introspection (see ``futureleak.libscan``).
Results go to ``results/indicator_audit.json`` and ``results/indicator_audit.md``.

One non-default setting is audited as well and reported separately: ``ta`` with
``fillna=True``, an option every ``ta`` indicator accepts.  It is not part of
the headline counts.

The whole audit is then repeated on a second synthetic data set with a
different seed, to see whether any verdict depends on the data.

A flagged indicator is only called confirmed when ``confirmed_causes.json`` has
an entry for the same library version, written after reading the source.
Anything else that is flagged is reported as unconfirmed.

Run from the repository root:

    python experiments/audit_indicator_libraries.py
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import platform
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import futureleak
from futureleak import synthetic_ohlcv
from futureleak.libscan import (
    LIBRARIES,
    ScanResult,
    horizon_text,
    library_version,
    scan_indicators,
    scan_library,
    summarise,
    ta_indicators,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CAUSES = Path(__file__).with_name("confirmed_causes.json")

N_ROWS = 600
DATA_SEED = 0
AUDIT_SEED = 0
N_CUTS = 40

# Second, independent run used only to check that the verdicts do not depend on the data.
REPEAT_DATA_SEED = 1
REPEAT_AUDIT_SEED = 1
REPEAT_N_CUTS = 20

UNCONFIRMED = "unconfirmed"
LOW_POWER = "little power"


def main() -> int:
    data = synthetic_ohlcv(N_ROWS, seed=DATA_SEED)
    causes = {
        k: v for k, v in json.loads(CAUSES.read_text(encoding="utf-8")).items() if k[0] != "_"
    }

    results: list[ScanResult] = []
    libraries: list[dict[str, Any]] = []
    not_audited: list[dict[str, str]] = []
    for name in LIBRARIES:
        try:
            scanned = scan_library(name, data, n_cuts=N_CUTS, seed=AUDIT_SEED)
        except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
            not_audited.append({"library": name, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"{name}: not audited ({type(exc).__name__}: {exc})")
            continue
        counts = summarise(scanned)
        counts["constant_output"] = sum(1 for r in scanned if _constant_output(r))
        libraries.append({"library": name, "version": scanned[0].version, **counts})
        results.extend(scanned)
        print(
            f"{name} {scanned[0].version}: found {counts['found']}, audited {counts['audited']}, "
            f"leak {counts['leak']}, errored {counts['error']}, skipped {counts['skipped']}"
        )
    not_audited.extend(_probe_other_libraries(data))

    flagged = [_flagged_entry(r, causes) for r in results if r.verdict == "leak"]
    variants = _scan_variants(data, causes)
    repeat = _repeat_run(results)
    totals = {
        key: sum(lib[key] for lib in libraries)
        for key in (
            "found",
            "audited",
            "causal",
            "leak",
            "inconclusive",
            "non_deterministic",
            "error",
            "skipped",
            "constant_output",
        )
    }
    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "futureleak": futureleak.__version__,
        },
        "parameters": {
            "rows": N_ROWS,
            "data_seed": DATA_SEED,
            "audit_seed": AUDIT_SEED,
            "n_cuts": N_CUTS,
            "data": "futureleak.synthetic_ohlcv(rows, seed=data_seed)",
        },
        "libraries": libraries,
        "totals": totals,
        "classification_counts": dict(Counter(f["classification"] for f in flagged)),
        "not_audited": not_audited,
        "flagged": flagged,
        "error_reasons": _reasons(results, "error"),
        "skip_reasons": _reasons(results, "skipped"),
        "variants": variants,
        "repeat_run": repeat,
        "results": [r.to_dict() for r in results],
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "indicator_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS / "indicator_audit.md").write_text(_markdown(payload, results), encoding="utf-8")
    print(
        f"total: found {totals['found']}, audited {totals['audited']}, leak {totals['leak']}, "
        f"errored {totals['error']}, skipped {totals['skipped']}"
    )
    for entry in flagged:
        print(f"  {entry['library']} {entry['indicator']}: {entry['classification']}")
    print(
        f"repeat run (data seed {REPEAT_DATA_SEED}): {repeat['compared']} indicators compared, "
        f"{len(repeat['changed'])} verdicts changed"
    )
    for variant in variants:
        counts = variant["counts"]
        print(
            f"variant {variant['name']}: audited {counts['audited']}, leak {counts['leak']}, "
            f"errored {counts['error']}, skipped {counts['skipped']}"
        )
        for entry in variant["flagged"]:
            print(f"  {entry['library']} {entry['indicator']}: {entry['classification']}")
    print(f"wrote {RESULTS / 'indicator_audit.json'}")
    print(f"wrote {RESULTS / 'indicator_audit.md'}")
    return 0


def _repeat_run(results: list[ScanResult]) -> dict[str, Any]:
    """Audit everything again on different data and list the outcomes that changed."""
    data = synthetic_ohlcv(N_ROWS, seed=REPEAT_DATA_SEED)
    first = {(r.library, r.indicator): r.verdict or r.status for r in results}
    second: dict[tuple[str, str], str] = {}
    for name in sorted({r.library for r in results}):
        for r in scan_library(name, data, n_cuts=REPEAT_N_CUTS, seed=REPEAT_AUDIT_SEED):
            second[(r.library, r.indicator)] = r.verdict or r.status
    changed = [
        {"library": lib, "indicator": ind, "first": outcome, "repeat": second.get((lib, ind))}
        for (lib, ind), outcome in first.items()
        if second.get((lib, ind)) != outcome
    ]
    return {
        "rows": N_ROWS,
        "data_seed": REPEAT_DATA_SEED,
        "audit_seed": REPEAT_AUDIT_SEED,
        "n_cuts": REPEAT_N_CUTS,
        "compared": len(first),
        "flagged": sorted(f"{lib}/{ind}" for (lib, ind), o in second.items() if o == "leak"),
        "changed": changed,
    }


def _scan_variants(data: pd.DataFrame, causes: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Non-default settings, kept apart from the default-parameter audit."""
    try:
        indicators, skipped = ta_indicators(fillna=True)
        version = library_version("ta")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return []
    scanned = scan_indicators(
        indicators, skipped, version=version, data=data, n_cuts=N_CUTS, seed=AUDIT_SEED
    )
    return [
        {
            "name": "ta with fillna=True",
            "library": "ta",
            "version": version,
            "counts": summarise(scanned),
            "flagged": [
                _flagged_entry(r, causes, prefix="ta[fillna=True]")
                for r in scanned
                if r.verdict == "leak"
            ],
            "results": [r.to_dict() for r in scanned],
        }
    ]


def _constant_output(result: ScanResult) -> bool:
    """Audited, not flagged, but every output column was constant on this data."""
    if result.report is None or result.verdict != "causal":
        return False
    return all(any(LOW_POWER in note for note in c.notes) for c in result.report.columns)


def _flagged_entry(
    result: ScanResult, causes: dict[str, dict[str, str]], prefix: str | None = None
) -> dict[str, Any]:
    assert result.report is not None
    columns = result.report.leaking_columns
    entry: dict[str, Any] = {
        "library": result.library,
        "version": result.version,
        "indicator": result.indicator,
        "call": result.call,
        "leaking_columns": [c.name for c in columns],
        "kinds": sorted({c.kind for c in columns if c.kind}),
        "horizon": horizon_text(result),
        "leak_fraction_min": min(c.leak_fraction for c in columns),
        "last_affected_row": max(
            (c.last_affected_row for c in columns if c.last_affected_row is not None), default=None
        ),
        "detail": result.detail,
    }
    known = causes.get(f"{prefix or result.library}/{result.indicator}")
    if known and known["version"] == result.version:
        entry.update(
            classification=known["classification"],
            cause=known["cause"],
            source=known["source"],
            documentation=known["documentation"],
        )
    else:
        reason = "no source confirmation recorded for this indicator"
        if known:
            reason = f"the recorded confirmation is for version {known['version']}"
        entry.update(classification=UNCONFIRMED, cause=reason, source="", documentation="")
    return entry


def _reasons(results: list[ScanResult], status: str) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = {}
    for r in results:
        if r.status == status:
            out.setdefault(r.library, Counter())[_shorten(r.detail)] += 1
    return {lib: dict(counter.most_common()) for lib, counter in out.items()}


def _shorten(text: str, limit: int = 90) -> str:
    # Some messages embed a whole array; keep them groupable by dropping its contents.
    text = re.sub(r"'\[.*?(\]'|$)", "'[...]'", text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _probe_other_libraries(data: pd.DataFrame) -> list[dict[str, str]]:
    """Libraries I tried that have no adapter, with the reason observed on this machine."""
    out: list[dict[str, str]] = []
    try:
        importlib.import_module("pandas_ta")
    except Exception as exc:  # noqa: BLE001 - the message is the result
        out.append(
            {"library": "pandas_ta", "reason": f"import failed: {type(exc).__name__}: {exc}"}
        )
    else:
        out.append({"library": "pandas_ta", "reason": "installed, but there is no adapter for it"})

    try:
        indicators = importlib.import_module("tti.indicators")
    except Exception as exc:  # noqa: BLE001
        out.append({"library": "tti", "reason": f"import failed: {type(exc).__name__}: {exc}"})
        return out
    version = importlib.metadata.version("tti")
    classes = [cls for _, cls in inspect.getmembers(indicators, inspect.isclass)]
    errors: Counter[str] = Counter()
    for cls in classes:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls(input_data=data.copy()).getTiData()
        except Exception as exc:  # noqa: BLE001
            errors[f"{type(exc).__name__}: {exc}"] += 1
    failed = sum(errors.values())
    if failed == len(classes):
        message, _ = errors.most_common(1)[0]
        out.append(
            {
                "library": "tti",
                "reason": f"version {version}: all {len(classes)} indicator classes raise on "
                f"this pandas with default arguments ({message})",
            }
        )
    else:
        out.append(
            {
                "library": "tti",
                "reason": f"version {version}: {failed} of {len(classes)} indicator classes "
                "raise with default arguments; there is no adapter for it",
            }
        )
    return out


# --- markdown -------------------------------------------------------------


def _markdown(payload: dict[str, Any], results: list[ScanResult]) -> str:
    env, params, totals = payload["environment"], payload["parameters"], payload["totals"]
    lines = [
        "# Look-ahead audit of technical-analysis libraries",
        "",
        "Generated by `experiments/audit_indicator_libraries.py`. Do not edit by hand.",
        "",
        f"- Python {env['python']} on {env['platform']}, pandas {env['pandas']}, "
        f"numpy {env['numpy']}, futureleak {env['futureleak']}",
        f"- Data: `synthetic_ohlcv({params['rows']}, seed={params['data_seed']})`, "
        f"{params['n_cuts']} cut points per indicator, audit seed {params['audit_seed']}",
        "- Every indicator is called with its default parameters.",
        "",
        "## Summary",
        "",
        "| library | version | found | audited | causal | leak | inconclusive | non-deterministic "
        "| errored | skipped |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lib in payload["libraries"]:
        lines.append(_summary_row(lib["library"], lib["version"], lib))
    lines.append(_summary_row("**total**", "", totals))
    lines += [
        "",
        "- found: indicators discovered by introspection.",
        "- audited: called successfully and tested at every cut point.",
        "- inconclusive: the output was entirely missing on this data, so nothing could be "
        "compared.",
        "- errored: the indicator raised on the full data with default parameters.",
        "- skipped: needs an input that is not an OHLCV column, has a required parameter "
        "without a default, or does not return one row per input row.",
        f"- {totals['constant_output']} of the {totals['causal']} indicators counted as causal "
        "returned a constant on this data (for example a candlestick pattern that never "
        "occurs), so the audit says little about them.",
        "",
        "## Flagged indicators",
        "",
    ]
    counts = payload["classification_counts"]
    if counts:
        lines.append(
            f"{len(payload['flagged'])} indicators were flagged: "
            + ", ".join(f"{n} {label}" for label, n in counts.items())
            + "."
        )
        lines += ["", *_flagged_table(payload["flagged"]), "", "### Causes", ""]
        lines += _cause_blocks(payload["flagged"])
    else:
        lines += ["No indicator was flagged.", ""]

    for variant in payload["variants"]:
        c = variant["counts"]
        lines += [
            f"## Non-default option: {variant['name']}",
            "",
            "This run is separate from the tables above and is not included in their counts.",
            "",
            f"{variant['library']} {variant['version']}: {c['found']} found, {c['audited']} "
            f"audited, {c['causal']} causal, {c['leak']} leak, {c['error']} errored, "
            f"{c['skipped']} skipped.",
            "",
        ]
        if variant["flagged"]:
            lines += [*_flagged_table(variant["flagged"]), ""]
            lines += _cause_blocks(variant["flagged"])

    rep = payload["repeat_run"]
    lines += [
        "## Repeat on a second data set",
        "",
        f"The default-parameter audit was repeated on `synthetic_ohlcv({rep['rows']}, "
        f"seed={rep['data_seed']})` with {rep['n_cuts']} cut points and audit seed "
        f"{rep['audit_seed']}. {rep['compared']} indicators were compared; "
        f"{len(rep['changed'])} had a different outcome.",
        "",
    ]
    if rep["changed"]:
        lines += ["| library | indicator | first run | repeat |", "|---|---|---|---|"]
        lines += [
            f"| {c['library']} | {c['indicator']} | {c['first']} | {c['repeat']} |"
            for c in rep["changed"]
        ]
        lines.append("")

    lines += ["## Libraries not audited", ""]
    if payload["not_audited"]:
        lines.extend(f"- `{item['library']}`: {item['reason']}" for item in payload["not_audited"])
    else:
        lines.append("None.")
    lines.append("")

    for title, key in (("Errors", "error_reasons"), ("Skipped", "skip_reasons")):
        lines += [f"## {title} by reason", "", "| library | count | reason |", "|---|---|---|"]
        for lib, reasons in payload[key].items():
            for reason, count in reasons.items():
                lines.append(f"| {lib} | {count} | {_cell(reason)} |")
        lines.append("")

    lines += [
        "## All indicators",
        "",
        "| library | version | indicator | verdict | horizon (rows) | notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        verdict = r.verdict if r.status == "audited" else r.status
        lines.append(
            f"| {r.library} | {r.version} | {r.indicator} | {verdict} | {horizon_text(r)} | "
            f"{_cell(_shorten(r.detail, 160))} |"
        )
    return "\n".join(lines) + "\n"


def _flagged_table(flagged: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| library | version | indicator | leaking outputs | kind | horizon (rows) "
        "| classification | source |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in flagged:
        lines.append(
            f"| {f['library']} | {f['version']} | {f['indicator']} | "
            f"{_cell(', '.join(f['leaking_columns']))} | {', '.join(f['kinds'])} | "
            f"{f['horizon']} | {f['classification']} | {_cell(f['source']) or '-'} |"
        )
    return lines


def _cause_blocks(flagged: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for f in flagged:
        lines.append(
            f"**{f['library']} {f['version']} `{f['indicator']}`** ({f['classification']})"
        )
        lines.append("")
        lines.append(f"- Call: `{f['call']}`")
        lines.append(f"- Measured: {f['detail']}")
        lines.append(f"- Cause: {f['cause']}")
        if f["source"]:
            lines.append(f"- Source: `{f['source']}`")
        if f["documentation"]:
            lines.append(f"- Documentation: {f['documentation']}")
        lines.append("")
    return lines


def _summary_row(name: str, version: str, counts: dict[str, Any]) -> str:
    return (
        f"| {name} | {version} | {counts['found']} | {counts['audited']} | {counts['causal']} | "
        f"{counts['leak']} | {counts['inconclusive']} | {counts['non_deterministic']} | "
        f"{counts['error']} | {counts['skipped']} |"
    )


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    sys.exit(main())

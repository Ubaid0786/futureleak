"""Audit every indicator of a technical-analysis library with its default parameters.

Indicators are found by introspection, not from a hand-written list, so the scan
follows whatever version of the library is installed.  Each adapter turns one
indicator into a function of an OHLCV frame (columns ``open``, ``high``,
``low``, ``close``, ``volume``).  Indicators that need an input other than OHLCV
or a parameter without a default are skipped with the reason recorded.

The libraries are optional; the ``indicators`` extra in ``pyproject.toml`` lists them.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .core import AlignmentError, AuditError, _as_frame, audit
from .report import LEAK, Report
from .synthetic import synthetic_ohlcv

OHLCV = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Indicator:
    """One auditable indicator: a function of an OHLCV frame."""

    library: str
    name: str
    call: str
    func: Callable[[pd.DataFrame], Any]


@dataclass(frozen=True)
class Skipped:
    """An indicator that was found but cannot be called with OHLCV data and defaults."""

    library: str
    name: str
    reason: str


@dataclass(frozen=True)
class ScanResult:
    """Outcome for one indicator.

    ``status`` is ``"audited"``, ``"error"`` (the indicator raised or returned
    nothing usable on the full data) or ``"skipped"``.
    """

    library: str
    version: str
    indicator: str
    status: str
    call: str = ""
    verdict: str | None = None
    detail: str = ""
    report: Report | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "library": self.library,
            "version": self.version,
            "indicator": self.indicator,
            "status": self.status,
            "call": self.call,
            "verdict": self.verdict,
            "detail": self.detail,
        }
        if self.report is not None:
            out["columns"] = [c.to_dict() for c in self.report.columns]
            out["truncation_errors"] = self.report.truncation_errors
            out["perturbation_errors"] = self.report.perturbation_errors
        return out


# --- adapters -------------------------------------------------------------


def _split_parameters(
    signature: inspect.Signature, aliases: dict[str, str]
) -> tuple[dict[str, str], str | None]:
    """Map OHLCV parameters to column names; report a required non-OHLCV parameter."""
    mapping: dict[str, str] = {}
    for name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if name in aliases:
            mapping[name] = aliases[name]
        elif param.default is param.empty:
            return mapping, name
    return mapping, None


def ta_indicators(fillna: bool = False) -> tuple[list[Indicator], list[Skipped]]:
    """``ta``: every indicator class; one output column per public result method.

    ``fillna=True`` passes the library's non-default ``fillna`` option to every class.
    """
    from ta.utils import IndicatorMixin

    aliases = {c: c for c in OHLCV}
    found: list[Indicator] = []
    skipped: list[Skipped] = []
    for module_name in ("momentum", "trend", "volatility", "volume", "others"):
        module = importlib.import_module(f"ta.{module_name}")
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or not issubclass(cls, IndicatorMixin):
                continue
            label = f"{module_name}.{cls_name}"
            mapping, missing = _split_parameters(inspect.signature(cls), aliases)
            if missing:
                skipped.append(
                    Skipped("ta", label, f"required parameter '{missing}' has no default")
                )
                continue
            methods = [
                name
                for name, member in inspect.getmembers(cls, inspect.isfunction)
                if not name.startswith("_")
                and name in vars(cls)
                and len(inspect.signature(member).parameters) == 1
            ]
            if not methods:
                skipped.append(Skipped("ta", label, "no public result methods"))
                continue

            options = {"fillna": True} if fillna else {}

            def func(
                df: pd.DataFrame, cls=cls, mapping=mapping, methods=methods, options=options
            ) -> pd.DataFrame:
                instance = cls(**{arg: df[col] for arg, col in mapping.items()}, **options)
                return pd.DataFrame({m: getattr(instance, m)() for m in methods})

            args = ", ".join([*mapping, *(f"{k}={v}" for k, v in options.items())])
            found.append(Indicator("ta", label, f"ta.{module_name}.{cls_name}({args})", func))
    return found, skipped


def _discover_finta() -> tuple[list[Indicator], list[Skipped]]:
    """``finta``: every public function on ``finta.TA``."""
    from finta import TA

    found: list[Indicator] = []
    skipped: list[Skipped] = []
    for name, member in inspect.getmembers(TA, inspect.isfunction):
        if name.startswith("_"):
            continue
        params = list(inspect.signature(member).parameters.values())
        missing = [p.name for p in params[1:] if p.default is p.empty]
        if missing:
            skipped.append(
                Skipped("finta", name, f"required parameter '{missing[0]}' has no default")
            )
            continue

        def func(df: pd.DataFrame, member=member) -> Any:
            return member(df)

        found.append(Indicator("finta", name, f"finta.TA.{name}(ohlcv)", func))
    return found, skipped


def _discover_stockstats() -> tuple[list[Indicator], list[Skipped]]:
    """``stockstats``: every column family registered in ``StockDataFrame.handler``."""
    import stockstats

    # The handler table lists some families twice: once with their column names
    # (``eribull``, ``eribear``) and once under a bare key (``eri``) that is not
    # itself a column.  Keep one entry per handler, the one with the most columns.
    families: dict[str, tuple[str, ...]] = {}
    for names, handler in stockstats.wrap(synthetic_ohlcv(5)).handler.items():
        current = families.get(handler.__name__, ())
        if len(names) > len(current):
            families[handler.__name__] = tuple(names)
    found: list[Indicator] = []
    for names in sorted(families.values()):

        def func(df: pd.DataFrame, names=names) -> pd.DataFrame:
            frame = stockstats.wrap(df.copy())
            return pd.DataFrame({name: frame[name] for name in names})

        columns = ", ".join(repr(n) for n in names)
        found.append(Indicator("stockstats", names[0], f"stockstats.wrap(ohlcv)[{columns}]", func))
    return found, []


def _discover_pandas_ta_classic() -> tuple[list[Indicator], list[Skipped]]:
    """``pandas_ta_classic``: every name in ``pandas_ta_classic.Category``."""
    import pandas_ta_classic as pta

    aliases = {c: c for c in OHLCV} | {"open_": "open"}
    found: list[Indicator] = []
    skipped: list[Skipped] = []
    for category, names in pta.Category.items():
        for name in names:
            label = f"{category}.{name}"
            member = getattr(pta, name, None)
            if not callable(member):
                skipped.append(Skipped("pandas_ta_classic", label, "not exported by the package"))
                continue
            mapping, missing = _split_parameters(inspect.signature(member), aliases)
            if missing:
                skipped.append(
                    Skipped(
                        "pandas_ta_classic",
                        label,
                        f"required parameter '{missing}' is not an OHLCV column",
                    )
                )
                continue

            def func(df: pd.DataFrame, member=member, mapping=mapping) -> Any:
                return member(**{arg: df[col] for arg, col in mapping.items()})

            args = ", ".join(mapping)
            found.append(
                Indicator("pandas_ta_classic", label, f"pandas_ta_classic.{name}({args})", func)
            )
    return found, skipped


def _discover_talib() -> tuple[list[Indicator], list[Skipped]]:
    """TA-Lib: every function reported by ``talib.get_functions()``, via the abstract API."""
    import talib
    from talib.abstract import Function  # type: ignore[attr-defined]

    found: list[Indicator] = []
    skipped: list[Skipped] = []
    for name in sorted(talib.get_functions()):
        function = Function(name)
        needed: list[str] = []
        for value in function.input_names.values():
            needed.extend([value] if isinstance(value, str) else list(value))
        extra = [col for col in needed if col not in OHLCV]
        if extra:
            skipped.append(
                Skipped("talib", name, f"required input '{extra[0]}' is not an OHLCV column")
            )
            continue

        def func(df: pd.DataFrame, name=name) -> Any:
            # A fresh Function per call: the abstract object caches inputs and outputs.
            return Function(name)(df.astype("float64"))

        found.append(Indicator("talib", name, f"talib.abstract.Function('{name}')(ohlcv)", func))
    return found, skipped


def _discover_btalib() -> tuple[list[Indicator], list[Skipped]]:
    """``bta-lib``: every indicator class returned by ``btalib.get_indicators()``."""
    import btalib
    from btalib.meta import metadata

    found: list[Indicator] = []
    skipped: list[Skipped] = []
    for cls in sorted(btalib.get_indicators(), key=lambda c: c.__name__):
        name = cls.__name__
        inputs = tuple(getattr(cls, "inputs", ()))
        extra = [col for col in inputs if col not in OHLCV]
        if extra:
            skipped.append(
                Skipped("btalib", name, f"required input '{extra[0]}' is not an OHLCV column")
            )
            continue

        def func(df: pd.DataFrame, cls=cls) -> pd.DataFrame:
            # bta-lib tracks nested indicator calls in a module-level stack and
            # does not unwind it when an indicator raises, after which every
            # later call fails.  Start each call from a clean stack.
            metadata.callstack.clear()
            return cls(df).df

        found.append(Indicator("btalib", name, f"btalib.{name}(ohlcv).df", func))
    return found, skipped


@dataclass(frozen=True)
class _Library:
    distribution: str
    discover: Callable[[], tuple[list[Indicator], list[Skipped]]]


LIBRARIES: dict[str, _Library] = {
    "ta": _Library("ta", ta_indicators),
    "finta": _Library("finta", _discover_finta),
    "stockstats": _Library("stockstats", _discover_stockstats),
    "pandas_ta_classic": _Library("pandas-ta-classic", _discover_pandas_ta_classic),
    "talib": _Library("TA-Lib", _discover_talib),
    "btalib": _Library("bta-lib", _discover_btalib),
}

_ALIASES = {"ta_lib": "talib", "bta_lib": "btalib"}


def resolve_library(name: str) -> str:
    """Accept import names and distribution names (``TA-Lib``, ``pandas-ta-classic``)."""
    key = name.strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in LIBRARIES:
        raise KeyError(f"unsupported library '{name}'; supported: {', '.join(sorted(LIBRARIES))}")
    return key


def library_version(library: str) -> str:
    return importlib.metadata.version(LIBRARIES[resolve_library(library)].distribution)


def discover(library: str) -> tuple[list[Indicator], list[Skipped]]:
    """Find the indicators of an installed library.

    Raises:
        KeyError: the library has no adapter.
        ImportError: the library is not installed or fails to import.
    """
    return LIBRARIES[resolve_library(library)].discover()


def scan_library(
    library: str,
    data: pd.DataFrame | None = None,
    *,
    n_cuts: int = 20,
    seed: int = 0,
) -> list[ScanResult]:
    """Audit every indicator of ``library`` on ``data`` (seeded synthetic bars by default)."""
    key = resolve_library(library)
    indicators, skipped = discover(key)
    return scan_indicators(
        indicators, skipped, version=library_version(key), data=data, n_cuts=n_cuts, seed=seed
    )


def scan_indicators(
    indicators: list[Indicator],
    skipped: list[Skipped] | None = None,
    *,
    version: str,
    data: pd.DataFrame | None = None,
    n_cuts: int = 20,
    seed: int = 0,
) -> list[ScanResult]:
    """Audit a list of indicators, as returned by :func:`discover`."""
    if data is None:
        data = synthetic_ohlcv(600, seed=seed)
    results = [
        ScanResult(item.library, version, item.name, "skipped", detail=item.reason)
        for item in skipped or []
    ]
    for indicator in indicators:
        results.append(_scan_one(indicator, version, data, n_cuts, seed))
    return sorted(results, key=lambda r: r.indicator.lower())


def _scan_one(
    indicator: Indicator, version: str, data: pd.DataFrame, n_cuts: int, seed: int
) -> ScanResult:
    def attempt(frame: pd.DataFrame) -> Report:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return audit(indicator.func, frame, n_cuts=n_cuts, seed=seed, name=indicator.call)

    prefix = ""
    report: Report | None
    try:
        report = attempt(data)
    except AuditError as first:
        # Several older libraries index positionally (``series[0]``), which
        # pandas 3 no longer allows on a date index.  Retry on integer labels so
        # that such indicators are still audited.
        report = None
        if not isinstance(data.index, pd.RangeIndex):
            renumbered = data.reset_index(drop=True)
            try:
                # On integer labels a short output (a table of price bins, say)
                # would pass for a slice of the input, so insist on one row per row.
                if _row_count(indicator, renumbered) == len(renumbered):
                    report = attempt(renumbered)
            except Exception:
                report = None
        if report is None:
            if isinstance(first, AlignmentError):
                return ScanResult(
                    indicator.library,
                    version,
                    indicator.name,
                    "skipped",
                    call=indicator.call,
                    detail="output is not one row per input row",
                )
            return ScanResult(
                indicator.library,
                version,
                indicator.name,
                "error",
                call=indicator.call,
                detail=_error_text(first),
            )
        prefix = f"audited on an integer index (on the date index: {_error_text(first)})"

    assert report is not None
    detail = "; ".join(part for part in (_describe(report), prefix) if part)
    return ScanResult(
        indicator.library,
        version,
        indicator.name,
        "audited",
        call=indicator.call,
        verdict=report.verdict,
        detail=detail,
        report=report,
    )


def _row_count(indicator: Indicator, frame: pd.DataFrame) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return len(_as_frame(indicator.func(frame.copy()))[0])


def _error_text(exc: AuditError) -> str:
    cause = exc.__cause__ or exc
    if isinstance(cause, AlignmentError):
        return "output is not one row per input row"
    message = _first_line(str(cause))
    return f"{type(cause).__name__}: {message}" if message else type(cause).__name__


def _describe(report: Report) -> str:
    """Short note: which columns leak and how, or why the audit is weak."""
    if report.verdict == LEAK:
        groups: dict[str, list[str]] = {}
        for c in report.leaking_columns:
            text = f"{c.kind}, horizon {c.horizon_text()}, {c.leak_fraction:.0%} of cuts"
            groups.setdefault(text, []).append(c.name)
        parts = [f"{', '.join(names)}: {text}" for text, names in groups.items()]
        first_cut = report.cuts[0]
        confined = [
            c.last_affected_row
            for c in report.leaking_columns
            if c.last_affected_row is not None and c.last_affected_row < first_cut
        ]
        if len(confined) == len(report.leaking_columns):
            parts.append(f"only rows up to position {max(confined)} are affected")
        return "; ".join(parts)
    notes = sorted({note for c in report.columns for note in c.notes})
    weak = [c.name for c in report.columns if c.notes]
    if notes and len(weak) == len(report.columns):
        return "; ".join(notes)
    if notes:
        return f"{', '.join(weak)}: " + "; ".join(notes)
    return ""


def _first_line(text: str, limit: int = 200) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


# --- rendering ------------------------------------------------------------


def summarise(results: list[ScanResult]) -> dict[str, int]:
    """Counts by outcome for one library's results."""
    counts = {
        "found": len(results),
        "audited": 0,
        "causal": 0,
        "leak": 0,
        "inconclusive": 0,
        "non_deterministic": 0,
        "error": 0,
        "skipped": 0,
    }
    for r in results:
        counts[r.status] += 1
        if r.verdict is not None:
            counts[r.verdict] += 1
    return counts


def horizon_text(result: ScanResult) -> str:
    """Horizons of the leaking columns, deduplicated."""
    if result.report is None:
        return "-"
    seen: list[str] = []
    for column in result.report.leaking_columns:
        text = column.horizon_text()
        if text not in seen:
            seen.append(text)
    return ", ".join(seen) if seen else "-"


def results_to_text(results: list[ScanResult]) -> str:
    if not results:
        return "no indicators found"
    counts = summarise(results)
    first = results[0]
    lines = [
        f"{first.library} {first.version}: {counts['found']} indicators found, "
        f"{counts['audited']} audited ({counts['causal']} causal, {counts['leak']} leak, "
        f"{counts['inconclusive']} inconclusive, {counts['non_deterministic']} "
        f"non-deterministic), {counts['error']} errored, {counts['skipped']} skipped"
    ]
    for title, keep in (
        ("leaks", lambda r: r.verdict == LEAK),
        ("could not be verified", lambda r: r.verdict in ("inconclusive", "non_deterministic")),
        ("errored", lambda r: r.status == "error"),
        ("skipped", lambda r: r.status == "skipped"),
    ):
        chosen = [r for r in results if keep(r)]
        if chosen:
            lines.append(f"{title}:")
            lines.extend(f"  {r.indicator}: {r.detail}" for r in chosen)
    return "\n".join(lines)


def results_to_markdown(results: list[ScanResult]) -> str:
    lines = [
        "| library | version | indicator | verdict | horizon (rows) | notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        verdict = r.verdict if r.status == "audited" else r.status
        notes = r.detail.replace("|", "\\|")
        horizon = horizon_text(r)
        lines.append(
            f"| {r.library} | {r.version} | {r.indicator} | {verdict} | {horizon} | {notes} |"
        )
    return "\n".join(lines) + "\n"

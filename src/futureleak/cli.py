"""Command line interface.

    futureleak audit package.module:function --data prices.csv --index-col date
    futureleak scan-lib ta

Exit status: 0 when nothing leaks, 1 when a leak is found, 2 on a usage or
runtime error, 3 when the function could not be verified (it is not
deterministic, or its output is entirely missing on the data).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .core import AuditError, audit
from .report import CAUSAL, LEAK
from .synthetic import synthetic_ohlcv

EXIT_OK, EXIT_LEAK, EXIT_ERROR, EXIT_UNVERIFIED = 0, 1, 2, 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        AuditError,
        ImportError,
        AttributeError,
        KeyError,
        OSError,
        ValueError,
        TypeError,
    ) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"futureleak: error: {message}", file=sys.stderr)
        return EXIT_ERROR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="futureleak",
        description="Detect look-ahead leakage in time-series feature functions.",
    )
    parser.add_argument("--version", action="version", version=f"futureleak {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    one = commands.add_parser("audit", help="audit one function")
    one.add_argument(
        "target",
        help="function to audit, as package.module:function or path/to/file.py:function",
    )
    one.add_argument("--data", help="CSV file with the input rows in time order")
    one.add_argument("--index-col", help="CSV column (name or position) to use as the index")
    one.add_argument("--column", help="pass only this column to the function, as a Series")
    one.add_argument("--rows", type=int, default=600, help="rows of synthetic data without --data")
    _add_common(one)
    one.set_defaults(handler=_run_audit)

    lib = commands.add_parser(
        "scan-lib", help="audit every indicator of a technical-analysis library"
    )
    lib.add_argument("library", help="ta, finta, stockstats, pandas_ta_classic, talib or btalib")
    lib.add_argument("--rows", type=int, default=600, help="rows of synthetic OHLCV data")
    _add_common(lib)
    lib.set_defaults(handler=_run_scan)
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-cuts", type=int, default=20, help="number of cut points")
    parser.add_argument("--seed", type=int, default=0, help="seed for cuts, noise and data")
    parser.add_argument("--rtol", type=float, default=1e-9, help="relative float tolerance")
    parser.add_argument("--atol", type=float, default=1e-12, help="absolute float tolerance")
    parser.add_argument(
        "--format", choices=("text", "json", "markdown"), default="text", help="output format"
    )
    parser.add_argument("--output", help="write the result to this file instead of stdout")


def _run_audit(args: argparse.Namespace) -> int:
    func = load_callable(args.target)
    if args.data:
        data: pd.DataFrame | pd.Series = load_csv(args.data, args.index_col)
    else:
        data = synthetic_ohlcv(args.rows, seed=args.seed)
    if args.column:
        if args.column not in data.columns:
            raise KeyError(f"column '{args.column}' not found; have {list(data.columns)}")
        data = data[args.column]

    report = audit(
        func,
        data,
        n_cuts=args.n_cuts,
        seed=args.seed,
        rtol=args.rtol,
        atol=args.atol,
        name=args.target,
    )
    if args.format == "json":
        text = report.to_json()
    elif args.format == "markdown":
        text = report.to_markdown()
    else:
        text = report.summary()
    _emit(text, args.output)
    if report.verdict == LEAK:
        return EXIT_LEAK
    return EXIT_OK if report.verdict == CAUSAL else EXIT_UNVERIFIED


def _run_scan(args: argparse.Namespace) -> int:
    from .libscan import results_to_markdown, results_to_text, scan_library

    data = synthetic_ohlcv(args.rows, seed=args.seed)
    results = scan_library(args.library, data, n_cuts=args.n_cuts, seed=args.seed)
    if args.format == "json":
        text = json.dumps([r.to_dict() for r in results], indent=2)
    elif args.format == "markdown":
        text = results_to_markdown(results)
    else:
        text = results_to_text(results)
    _emit(text, args.output)
    return EXIT_LEAK if any(r.verdict == LEAK for r in results) else EXIT_OK


def load_callable(target: str) -> Callable[..., Any]:
    """Resolve ``package.module:attr`` or ``path/to/file.py:attr`` to a callable."""
    module_part, sep, attr_part = target.rpartition(":")
    if not sep or not module_part or not attr_part:
        raise ValueError(f"expected module:function, got '{target}'")
    if module_part.endswith(".py"):
        path = Path(module_part).resolve()
        if not path.is_file():
            raise ImportError(f"no such file: {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(path.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(path.parent))
    else:
        if "" not in sys.path and str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))
        module = importlib.import_module(module_part)
    obj: Any = module
    for name in attr_part.split("."):
        obj = getattr(obj, name)
    if not callable(obj):
        raise TypeError(f"'{target}' is not callable")
    return obj


def load_csv(path: str, index_col: str | None) -> pd.DataFrame:
    """Read a CSV; an index column that looks like dates becomes a DatetimeIndex."""
    column: str | int | None = index_col
    if index_col is not None and index_col.lstrip("-").isdigit():
        column = int(index_col)
    frame = pd.read_csv(path, index_col=column)
    if column is not None and not pd.api.types.is_numeric_dtype(frame.index.dtype):
        try:
            frame.index = pd.to_datetime(frame.index)
        except (ValueError, TypeError):
            pass
    return frame


def _emit(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())

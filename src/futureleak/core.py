"""The audit itself: truncation and perturbation tests at random cut points."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .perturb import perturb
from .report import (
    CAUSAL,
    FIXED_HORIZON,
    FULL_SAMPLE,
    INCONCLUSIVE,
    LEAK,
    NON_DETERMINISTIC,
    TRUNCATION_ONLY,
    VARIABLE_HORIZON,
    ColumnReport,
    Report,
)

Feature = Callable[[Any], Any]
"""A callable taking a DataFrame or Series and returning features on the same index."""


class AuditError(Exception):
    """The function could not be audited at all (it raised on the full input, for example)."""


class AlignmentError(AuditError):
    """The function's output could not be aligned with the rows of its input."""


def audit(
    func: Feature,
    data: pd.DataFrame | pd.Series,
    *,
    n_cuts: int = 20,
    seed: int = 0,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    min_cut: int | None = None,
    name: str | None = None,
) -> Report:
    """Test whether ``func(data)`` at row ``t`` depends on rows after ``t``.

    For each of ``n_cuts`` cut positions ``t`` the function is called on the
    data truncated after ``t`` and on the data with every row after ``t``
    perturbed.  Both results must agree with ``func(data)`` on rows up to and
    including ``t``.  When the perturbed result differs, a search over ``h``
    finds how far ahead the function reads.

    Args:
        func: Callable taking the same type as ``data`` and returning a Series,
            DataFrame, array, or a tuple or mapping of those.  A pandas result
            may be shorter than the input as long as it keeps the input's index
            labels; an array must have one row per input row.
        data: Input rows in time order.
        n_cuts: Number of cut points.
        seed: Seed for cut selection and for the perturbation.
        rtol: Relative float tolerance.
        atol: Absolute float tolerance.
        min_cut: Earliest row position allowed as a cut.  Defaults to a quarter
            of the way into the data so that warm-up periods are skipped.
        name: Label for the report; defaults to the function's own name.

    Returns:
        A :class:`~futureleak.report.Report` with one entry per output column.

    Raises:
        AuditError: ``func`` raised or returned nothing on the full input.
        AlignmentError: the output could not be matched to input rows.
        TypeError, ValueError: the arguments are unusable.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError(f"data must be a pandas DataFrame or Series, got {type(data).__name__}")
    n = len(data)
    if n < 4:
        raise ValueError(f"data needs at least 4 rows, got {n}")
    if n_cuts < 1:
        raise ValueError("n_cuts must be at least 1")

    label = name or _callable_name(func)
    rng = np.random.default_rng(seed)
    cuts = _choose_cuts(n, n_cuts, rng, min_cut)

    try:
        base_frame = _normalise(func(data.copy(deep=True)), data.index)
        repeat_frame = _normalise(func(data.copy(deep=True)), data.index)
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError(f"{label} raised {type(exc).__name__} on the full input: {exc}") from exc

    columns = list(base_frame.columns)
    base = _to_arrays(base_frame, columns)
    repeat = _to_arrays(repeat_frame, columns)
    unstable = {
        col for col in columns if _difference(base[col], repeat[col], rtol, atol) is not None
    }

    perturbed = perturb(data, rng)

    def mixed(last_original: int) -> pd.DataFrame | pd.Series:
        """Original rows up to ``last_original``, perturbed rows after it."""
        if last_original >= n - 1:
            return data.copy(deep=True)
        out = pd.concat([data.iloc[: last_original + 1], perturbed.iloc[last_original + 1 :]])
        out.index = data.index  # concat can drop index attributes such as freq
        return out

    def run(frame: pd.DataFrame | pd.Series) -> dict[str, np.ndarray] | None:
        try:
            return _to_arrays(_normalise(func(frame), frame.index), columns)
        except Exception:
            return None

    def run_shifted(t: int, h: int) -> dict[str, np.ndarray] | None:
        """Outputs when only rows after ``t + h`` are perturbed."""
        return run(mixed(t + h))

    stats = {col: _ColumnStats() for col in columns}
    truncation_ok = perturbation_ok = evaluated = 0

    for t in cuts:
        truncated = run(data.iloc[: t + 1].copy(deep=True))
        disturbed = run(mixed(t))
        if truncated is not None:
            truncation_ok += 1
        if disturbed is not None:
            perturbation_ok += 1
        if truncated is None and disturbed is None:
            continue
        evaluated += 1

        remaining = n - 1 - t
        probe = _Probe(functools.partial(run_shifted, t), base, t, rtol, atol)

        for col in columns:
            if col in unstable:
                continue
            st = stats[col]
            hit = False
            if truncated is not None:
                diff = _difference(base[col][: t + 1], truncated[col], rtol, atol)
                if diff is not None:
                    hit = True
                    st.truncation_leaks += 1
                    st.record(diff, t)
            if disturbed is not None:
                diff = _difference(base[col][: t + 1], disturbed[col][: t + 1], rtol, atol)
                if diff is not None:
                    hit = True
                    st.perturbation_leaks += 1
                    st.record(diff, t)
                    horizon = _search_horizon(col, remaining, probe)
                    if horizon is not None:
                        st.horizons.append((horizon, horizon >= remaining, remaining))
            if hit:
                st.leaking_cuts += 1

    reports = tuple(
        _column_report(
            col,
            stats[col],
            base[col],
            unstable=col in unstable,
            first_cut=cuts[0],
            evaluated=evaluated,
            truncation_ok=truncation_ok,
            perturbation_ok=perturbation_ok,
        )
        for col in columns
    )
    notes: list[str] = []
    if unstable:
        notes.append(
            "Two calls on identical input returned different values, so the affected "
            "columns cannot be tested for leakage."
        )
    return Report(
        function=label,
        n_rows=n,
        cuts=tuple(cuts),
        seed=seed,
        rtol=rtol,
        atol=atol,
        columns=reports,
        truncation_errors=len(cuts) - truncation_ok,
        perturbation_errors=len(cuts) - perturbation_ok,
        notes=tuple(notes),
    )


@dataclass
class _ColumnStats:
    truncation_leaks: int = 0
    perturbation_leaks: int = 0
    leaking_cuts: int = 0
    # (horizon, reached end of data, rows remaining after the cut)
    horizons: list[tuple[int, bool, int]] = field(default_factory=list)
    earliest_offset: int | None = None
    last_affected_row: int | None = None
    max_abs_diff: float | None = None

    def record(self, diff: _Difference, cut: int) -> None:
        offset = diff.first - cut
        if self.earliest_offset is None or offset < self.earliest_offset:
            self.earliest_offset = offset
        if self.last_affected_row is None or diff.last > self.last_affected_row:
            self.last_affected_row = diff.last
        if diff.size is not None and (self.max_abs_diff is None or diff.size > self.max_abs_diff):
            self.max_abs_diff = diff.size


class _Probe:
    """Which columns change up to cut ``t`` when only rows after ``t + h`` are perturbed.

    One call of the audited function answers the question for every column, so
    the answers are cached by ``h`` and shared between columns.
    """

    def __init__(
        self,
        run_shifted: Callable[[int], dict[str, np.ndarray] | None],
        base: dict[str, np.ndarray],
        t: int,
        rtol: float,
        atol: float,
    ) -> None:
        self._run_shifted = run_shifted
        self._base = base
        self._t = t
        self._rtol = rtol
        self._atol = atol
        self._cache: dict[int, dict[str, bool] | None] = {}

    def __call__(self, h: int) -> dict[str, bool] | None:
        """``None`` means the function raised on that input."""
        if h not in self._cache:
            result = self._run_shifted(h)
            stop = self._t + 1
            self._cache[h] = (
                None
                if result is None
                else {
                    col: _difference(values[:stop], result[col][:stop], self._rtol, self._atol)
                    is not None
                    for col, values in self._base.items()
                }
            )
        return self._cache[h]


def _search_horizon(col: str, remaining: int, probe: _Probe) -> int | None:
    """Smallest ``h`` at which perturbing rows after ``t + h`` no longer changes ``col``.

    ``h = 0`` is already known to leak and ``h = remaining`` perturbs nothing.
    Dependence on a set of future rows makes the leak indicator monotone in
    ``h``, which allows a galloping search followed by bisection.
    """

    def leaks(h: int) -> bool | None:
        result = probe(h)
        return None if result is None else result[col]

    if remaining <= 1:
        return remaining
    # One probe on the last row alone settles the common full-sample case.
    last = leaks(remaining - 1)
    if last is None:
        return None
    if last:
        return remaining

    low, high = 0, remaining - 1  # leaks at low, does not leak at high
    step = 1
    while step < high:
        state = leaks(step)
        if state is None:
            return None
        if state:
            low = step
            step *= 2
        else:
            high = step
            break
    while high - low > 1:
        mid = (low + high) // 2
        state = leaks(mid)
        if state is None:
            return None
        if state:
            low = mid
        else:
            high = mid
    return high


def _column_report(
    col: str,
    st: _ColumnStats,
    base: np.ndarray,
    *,
    unstable: bool,
    first_cut: int,
    evaluated: int,
    truncation_ok: int,
    perturbation_ok: int,
) -> ColumnReport:
    if unstable:
        return ColumnReport(
            name=col,
            verdict=NON_DETERMINISTIC,
            notes=("two calls on the same input disagree",),
        )

    notes: list[str] = []
    missing = pd.isna(base)
    all_missing = bool(missing.all())
    if not all_missing:
        present = base[~missing]
        if len(present) > 1 and all(_same(v, present[0]) for v in present[1:]):
            notes.append("output is constant on this data, so the audit has little power")

    truncation_fraction = st.truncation_leaks / truncation_ok if truncation_ok else None
    perturbation_fraction = st.perturbation_leaks / perturbation_ok if perturbation_ok else None

    if st.leaking_cuts == 0:
        if evaluated == 0:
            verdict = INCONCLUSIVE
            notes.append("the function raised at every cut point")
        elif all_missing:
            verdict = INCONCLUSIVE
            notes.append("output is entirely missing on this data")
        else:
            verdict = CAUSAL
        return ColumnReport(
            name=col,
            verdict=verdict,
            truncation_leak_fraction=truncation_fraction,
            perturbation_leak_fraction=perturbation_fraction,
            notes=tuple(notes),
        )

    kind, horizon, horizon_min, to_end = _classify(st)
    if kind == TRUNCATION_ONLY:
        notes.append(
            "changes when later rows are removed but not when their values change: "
            "the output depends on the length of the input"
        )
    elif horizon is None:
        notes.append("the horizon could not be measured: the function raised during the search")
    if st.last_affected_row is not None and st.last_affected_row < first_cut:
        notes.append(
            f"only rows up to position {st.last_affected_row} are affected: the start of "
            "the series depends on later data, the rest does not"
        )
    return ColumnReport(
        name=col,
        verdict=LEAK,
        kind=kind,
        leak_fraction=st.leaking_cuts / evaluated,
        horizon=horizon,
        horizon_min=horizon_min,
        horizon_to_end=to_end,
        earliest_offset=st.earliest_offset,
        last_affected_row=st.last_affected_row,
        truncation_leak_fraction=truncation_fraction,
        perturbation_leak_fraction=perturbation_fraction,
        max_abs_diff=st.max_abs_diff,
        notes=tuple(notes),
    )


def _classify(st: _ColumnStats) -> tuple[str, int | None, int | None, bool]:
    """Return (kind, horizon, smallest horizon, horizon reached end of data)."""
    if st.perturbation_leaks == 0:
        return TRUNCATION_ONLY, None, None, False
    if not st.horizons:
        return VARIABLE_HORIZON, None, None, False
    bounded = [h for h, to_end, _ in st.horizons if not to_end]
    censored = [rows for _, to_end, rows in st.horizons if to_end]
    if not bounded:
        return FULL_SAMPLE, max(censored), min(censored), True
    low, high = min(bounded), max(bounded)
    # A cut too close to the end cannot show the whole horizon; it is consistent
    # with a fixed horizon as long as it had no more rows left than that horizon.
    if low == high and all(rows <= high for rows in censored):
        return FIXED_HORIZON, high, high, False
    return VARIABLE_HORIZON, max(high, *censored) if censored else high, low, False


def _choose_cuts(n: int, n_cuts: int, rng: np.random.Generator, min_cut: int | None) -> list[int]:
    """One random cut in each of ``n_cuts`` equal slices of the allowed range."""
    low = max(1, n // 4) if min_cut is None else min_cut
    high = n - 2  # at least one row must remain after the cut
    if low < 0 or low > high:
        raise ValueError(f"min_cut must lie in [0, {high}] for {n} rows, got {low}")
    if n_cuts >= high - low + 1:
        return list(range(low, high + 1))
    edges = np.linspace(low, high + 1, n_cuts + 1)
    cuts = set()
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        a = int(math.floor(start))
        b = max(a + 1, int(math.floor(stop)))
        cuts.add(int(rng.integers(a, b)))
    return sorted(cuts)


def _callable_name(func: Any) -> str:
    if isinstance(func, functools.partial):
        return _callable_name(func.func)
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    if qualname is None:
        return type(func).__name__
    if module and module not in ("__main__", "builtins"):
        return f"{module}.{qualname}"
    return qualname


# --- output normalisation -------------------------------------------------


def _normalise(out: Any, index: pd.Index) -> pd.DataFrame:
    """Turn any supported return value into a DataFrame on ``index``."""
    frame, labelled = _as_frame(out)
    frame = frame.copy()
    frame.columns = _unique_names(frame.columns)
    return _align(frame, index, labelled)


def _as_frame(out: Any) -> tuple[pd.DataFrame, bool]:
    """Return the output as a frame and whether its index carries row labels."""
    if out is None:
        raise AuditError("the function returned None")
    if isinstance(out, pd.DataFrame):
        return out, True
    if isinstance(out, pd.Series):
        return out.to_frame(name="value" if out.name is None else out.name), True
    if isinstance(out, np.ndarray):
        if out.ndim == 1:
            return pd.DataFrame({"value": out}), False
        if out.ndim == 2:
            return pd.DataFrame(out), False
        raise AuditError(f"cannot audit an array with {out.ndim} dimensions")
    if isinstance(out, Mapping):
        return _combine([(str(key), value) for key, value in out.items()])
    if isinstance(out, Sequence) and not isinstance(out, (str, bytes)):
        if out and all(np.isscalar(v) or v is None for v in out):
            return pd.DataFrame({"value": np.asarray(out)}), False
        return _combine([(f"out{i}", value) for i, value in enumerate(out)])
    raise AuditError(f"cannot audit a return value of type {type(out).__name__}")


def _combine(parts: list[tuple[str, Any]]) -> tuple[pd.DataFrame, bool]:
    frames: list[pd.DataFrame] = []
    labelled = True
    for key, value in parts:
        frame, has_labels = _as_frame(value)
        labelled = labelled and has_labels
        if frame.shape[1] == 1:
            frame = frame.set_axis([key], axis=1)
        else:
            frame = frame.set_axis([f"{key}.{c}" for c in frame.columns], axis=1)
        frames.append(frame)
    if not frames:
        raise AuditError("the function returned an empty collection")
    if labelled:
        return pd.concat(frames, axis=1), True
    # Arrays carry no labels, so the parts can only be matched by position.
    if len({len(frame) for frame in frames}) != 1:
        raise AlignmentError("the function returned unlabelled outputs of different lengths")
    index = next(
        (f.index for f, (_, v) in zip(frames, parts, strict=True) if _as_frame(v)[1]), None
    )
    combined = pd.concat([frame.reset_index(drop=True) for frame in frames], axis=1)
    if index is None:
        return combined, False
    return combined.set_axis(index, axis=0), True


def _unique_names(columns: pd.Index) -> list[str]:
    seen: dict[str, int] = {}
    names = []
    for col in columns:
        text = str(col)
        count = seen.get(text, 0)
        seen[text] = count + 1
        names.append(text if count == 0 else f"{text}#{count}")
    return names


def _align(frame: pd.DataFrame, index: pd.Index, labelled: bool) -> pd.DataFrame:
    if frame.index.equals(index):
        return frame
    if labelled and index.is_unique and frame.index.is_unique:
        try:
            subset = bool(frame.index.isin(index).all())
        except TypeError:
            subset = False
        # Rows the function dropped (warm-up, dropna) come back as missing.
        if subset:
            return frame.reindex(index)
    if len(frame) == len(index):
        return frame.set_axis(index, axis=0)
    raise AlignmentError(
        f"the output has {len(frame)} rows for {len(index)} input rows and its index is "
        "not a subset of the input index, so rows cannot be matched. Return a pandas "
        "object that keeps the input's index labels."
    )


# --- comparison -----------------------------------------------------------


def _to_arrays(frame: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    """One float array per numeric column, one object array per other column."""
    out: dict[str, np.ndarray] = {}
    for col in columns:
        if col not in frame.columns:
            out[col] = np.full(len(frame), np.nan)
            continue
        series = frame[col]
        dtype = series.dtype
        numeric = (
            pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_numeric_dtype(dtype)
            and not pd.api.types.is_complex_dtype(dtype)
        )
        if numeric:
            out[col] = series.to_numpy(dtype="float64", na_value=np.nan)
        elif pd.api.types.is_object_dtype(dtype):
            try:
                out[col] = series.to_numpy(dtype="float64", na_value=np.nan)
            except (TypeError, ValueError):
                out[col] = series.to_numpy(dtype=object)
        else:
            out[col] = series.astype(object).to_numpy(dtype=object)
    return out


@dataclass(frozen=True)
class _Difference:
    first: int  # position of the first differing row
    last: int  # position of the last differing row
    size: float | None  # largest finite absolute difference, numeric columns only


def _difference(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> _Difference | None:
    """Where two columns differ, or ``None`` if they agree.

    Missing values compare equal to each other and unequal to anything else.
    """
    length = min(len(a), len(b))
    a, b = a[:length], b[:length]
    if a.dtype == np.float64 and b.dtype == np.float64:
        with np.errstate(invalid="ignore"):
            gap = np.abs(a - b)
            close = gap <= atol + rtol * np.maximum(np.abs(a), np.abs(b))
        same = close | (a == b) | (np.isnan(a) & np.isnan(b))
        differs = ~same
        if not differs.any():
            return None
        finite = gap[differs & np.isfinite(gap)]
        return _span(differs, float(finite.max()) if finite.size else None)

    a_obj, b_obj = a.astype(object), b.astype(object)
    differs = np.fromiter(
        (not _same(x, y) for x, y in zip(a_obj, b_obj, strict=True)), dtype=bool, count=length
    )
    if not differs.any():
        return None
    return _span(differs, None)


def _span(differs: np.ndarray, size: float | None) -> _Difference:
    first = int(np.argmax(differs))
    last = len(differs) - 1 - int(np.argmax(differs[::-1]))
    return _Difference(first, last, size)


def _same(x: Any, y: Any) -> bool:
    x_missing, y_missing = _is_missing(x), _is_missing(y)
    if x_missing or y_missing:
        return x_missing and y_missing
    try:
        return bool(x == y)
    except Exception:
        return False


def _is_missing(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False

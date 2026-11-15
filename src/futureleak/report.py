"""Result containers for an audit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Verdicts for one output column.
CAUSAL = "causal"
LEAK = "leak"
NON_DETERMINISTIC = "non_deterministic"
INCONCLUSIVE = "inconclusive"

# Leak kinds.
#
# A fixed-horizon leak reads a bounded number of rows ahead, the same number at
# every cut (``shift(-k)``, a centred window).  Lagging the output by the
# horizon makes it causal.  The other three kinds are truncation-sensitive: the
# value at ``t`` depends on how much data exists after ``t``, so no lag repairs
# them and the feature has to be recomputed on data up to ``t`` only.
FIXED_HORIZON = "fixed_horizon"
FULL_SAMPLE = "full_sample"
VARIABLE_HORIZON = "variable_horizon"
TRUNCATION_ONLY = "truncation_only"

TRUNCATION_SENSITIVE_KINDS = frozenset({FULL_SAMPLE, VARIABLE_HORIZON, TRUNCATION_ONLY})


@dataclass(frozen=True)
class ColumnReport:
    """Audit result for one output column.

    Attributes:
        name: Column name (``"value"`` for an unnamed Series).
        verdict: ``"causal"``, ``"leak"``, ``"non_deterministic"`` or
            ``"inconclusive"``.
        kind: For leaks, one of ``"fixed_horizon"``, ``"full_sample"``,
            ``"variable_horizon"``, ``"truncation_only"``; otherwise ``None``.
        leak_fraction: Share of evaluated cut points at which a leak was seen.
        horizon: Look-ahead horizon in rows: the smallest ``h`` such that
            perturbing only rows after ``t + h`` leaves every output at or
            before ``t`` unchanged, maximised over cut points.  ``None`` when
            there is no leak or when only the truncation test fired.
        horizon_min: Smallest horizon seen across leaking cut points.
        horizon_to_end: ``True`` when the horizon ran to the last row of the
            data at every leaking cut, so ``horizon`` is only a lower bound.
        earliest_offset: Most negative ``row - t`` among affected rows.  ``0``
            means only the row at the cut changed; ``-5`` means the leak
            reached five rows back from the cut.
        last_affected_row: Largest row position at which a difference was seen.
            A value below every cut point means that only the start of the
            series is contaminated.
        truncation_leak_fraction: Share of cuts where ``f(data[:t])`` differed
            from ``f(data)[:t]``.
        perturbation_leak_fraction: Share of cuts where perturbing rows after
            ``t`` changed outputs at or before ``t``.
        max_abs_diff: Largest absolute difference observed in a numeric column.
        notes: Free-text remarks (low power, errors, and so on).
    """

    name: str
    verdict: str
    kind: str | None = None
    leak_fraction: float = 0.0
    horizon: int | None = None
    horizon_min: int | None = None
    horizon_to_end: bool = False
    earliest_offset: int | None = None
    last_affected_row: int | None = None
    truncation_leak_fraction: float | None = None
    perturbation_leak_fraction: float | None = None
    max_abs_diff: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def leaks(self) -> bool:
        return self.verdict == LEAK

    @property
    def truncation_sensitive(self) -> bool:
        """True for leaks that cannot be repaired by lagging the output."""
        return self.kind in TRUNCATION_SENSITIVE_KINDS

    def horizon_text(self) -> str:
        if self.horizon is None:
            return "-"
        if self.horizon_to_end:
            return f">={self.horizon} (to end of data)"
        if self.horizon_min is not None and self.horizon_min != self.horizon:
            return f"{self.horizon_min}..{self.horizon}"
        return str(self.horizon)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["notes"] = list(self.notes)
        return out


@dataclass(frozen=True)
class Report:
    """Audit result for one function.

    Attributes:
        function: Name of the audited callable.
        n_rows: Number of input rows.
        cuts: Row positions used as cut points.
        seed: Seed for cut selection and perturbation.
        rtol: Relative tolerance used when comparing floats.
        atol: Absolute tolerance used when comparing floats.
        columns: One :class:`ColumnReport` per output column.
        truncation_errors: Cut points at which the function raised on the
            truncated input.
        perturbation_errors: Cut points at which the function raised on the
            perturbed input.
        notes: Remarks that apply to the whole audit.
    """

    function: str
    n_rows: int
    cuts: tuple[int, ...]
    seed: int
    rtol: float
    atol: float
    columns: tuple[ColumnReport, ...]
    truncation_errors: int = 0
    perturbation_errors: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def leaks(self) -> bool:
        """True if any output column leaks."""
        return any(c.leaks for c in self.columns)

    @property
    def leaking_columns(self) -> tuple[ColumnReport, ...]:
        return tuple(c for c in self.columns if c.leaks)

    @property
    def non_deterministic(self) -> bool:
        return any(c.verdict == NON_DETERMINISTIC for c in self.columns)

    @property
    def inconclusive(self) -> bool:
        return any(c.verdict == INCONCLUSIVE for c in self.columns)

    @property
    def verdict(self) -> str:
        """Worst verdict across columns: leak > non_deterministic > inconclusive > causal."""
        if self.leaks:
            return LEAK
        if self.non_deterministic:
            return NON_DETERMINISTIC
        if self.inconclusive:
            return INCONCLUSIVE
        return CAUSAL

    def __getitem__(self, column: str) -> ColumnReport:
        for c in self.columns:
            if c.name == column:
                return c
        raise KeyError(column)

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "verdict": self.verdict,
            "n_rows": self.n_rows,
            "cuts": list(self.cuts),
            "seed": self.seed,
            "rtol": self.rtol,
            "atol": self.atol,
            "truncation_errors": self.truncation_errors,
            "perturbation_errors": self.perturbation_errors,
            "notes": list(self.notes),
            "columns": [c.to_dict() for c in self.columns],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f"### futureleak audit: `{self.function}`",
            "",
            f"Verdict: **{self.verdict}**. {self.n_rows} rows, {len(self.cuts)} cut points, "
            f"seed {self.seed}.",
            "",
            "| column | verdict | kind | leaking cuts | horizon (rows) | earliest offset | notes |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in self.columns:
            offset = "-" if c.earliest_offset is None else str(c.earliest_offset)
            lines.append(
                f"| {_cell(c.name)} | {c.verdict} | {c.kind or '-'} | {c.leak_fraction:.0%} | "
                f"{c.horizon_text()} | {offset} | {_cell('; '.join(c.notes))} |"
            )
        extra = list(self.notes)
        if self.truncation_errors:
            extra.append(
                f"The function raised on the truncated input at {self.truncation_errors} "
                f"of {len(self.cuts)} cut points."
            )
        if self.perturbation_errors:
            extra.append(
                f"The function raised on the perturbed input at {self.perturbation_errors} "
                f"of {len(self.cuts)} cut points."
            )
        if extra:
            lines.append("")
            lines.extend(f"- {note}" for note in extra)
        return "\n".join(lines) + "\n"

    def summary(self) -> str:
        """Plain-text summary, one line per column."""
        lines = [f"{self.function}: {self.verdict} ({self.n_rows} rows, {len(self.cuts)} cuts)"]
        for c in self.columns:
            line = f"  {c.name}: {c.verdict}"
            if c.leaks:
                line += (
                    f" [{c.kind}] leaking cuts {c.leak_fraction:.0%}, "
                    f"horizon {c.horizon_text()}, earliest offset {c.earliest_offset}"
                )
            if c.notes:
                line += " (" + "; ".join(c.notes) + ")"
            lines.append(line)
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")

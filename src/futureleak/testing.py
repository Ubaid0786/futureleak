"""Assertion helper for test suites."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .core import Feature, audit
from .report import Report


class LeakageError(AssertionError):
    """Raised by :func:`assert_causal`.  The full report is on ``.report``."""

    def __init__(self, message: str, report: Report) -> None:
        super().__init__(message)
        self.report = report


def assert_causal(
    func: Feature,
    data: pd.DataFrame | pd.Series,
    *,
    allow_non_deterministic: bool = False,
    allow_inconclusive: bool = False,
    **audit_kwargs: Any,
) -> Report:
    """Fail unless ``func`` is shown to use no rows after ``t`` for its output at ``t``.

    Intended for pytest::

        def test_features_are_causal(prices):
            assert_causal(build_features, prices)

    A function that cannot be verified also fails by default: one that returns
    different values on identical input, or whose output is entirely missing on
    this data.  Pass ``allow_non_deterministic`` or ``allow_inconclusive`` to
    accept those cases.  Other keyword arguments go to :func:`futureleak.audit`.

    Returns:
        The report, when the assertion holds.

    Raises:
        LeakageError: a column leaks or could not be verified.
    """
    report = audit(func, data, **audit_kwargs)
    if report.leaks:
        names = ", ".join(c.name for c in report.leaking_columns)
        raise LeakageError(
            f"look-ahead leakage in {report.function}: {names}\n{report.summary()}", report
        )
    if report.non_deterministic and not allow_non_deterministic:
        raise LeakageError(
            f"{report.function} is not deterministic, so causality cannot be verified\n"
            f"{report.summary()}",
            report,
        )
    if report.inconclusive and not allow_inconclusive:
        raise LeakageError(
            f"the audit of {report.function} was inconclusive\n{report.summary()}", report
        )
    return report

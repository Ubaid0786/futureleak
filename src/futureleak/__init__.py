"""Look-ahead leakage auditor for time-series feature pipelines.

A feature computed at time ``t`` must not change when data after ``t`` changes.
:func:`audit` tests that property by truncating and perturbing the input, with
no knowledge of how the function works.
"""

from .core import AlignmentError, AuditError, audit
from .report import ColumnReport, Report
from .synthetic import synthetic_ohlcv
from .testing import LeakageError, assert_causal

__version__ = "0.1.0"

__all__ = [
    "AlignmentError",
    "AuditError",
    "ColumnReport",
    "LeakageError",
    "Report",
    "assert_causal",
    "audit",
    "synthetic_ohlcv",
    "__version__",
]

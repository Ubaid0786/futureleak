"""Perturbation of input data.

The audit needs a second version of the input that differs from the original in
every row.  Rows up to a cut are then taken from the original and rows after it
from the perturbed copy, so a causal function must return the same values up to
the cut.

The perturbation is deliberately large (log-normal factors with unit sigma for
positive columns) so that future extremes exceed past extremes.  A function
that depends on the full-sample maximum or minimum is then caught as well as
one that depends on the full-sample mean.  The perturbed values do not have to
be plausible: any dependence of a past output on a future input is a leak,
whatever the future input looks like.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd


@overload
def perturb(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame: ...


@overload
def perturb(data: pd.Series, rng: np.random.Generator) -> pd.Series: ...


def perturb(data: pd.DataFrame | pd.Series, rng: np.random.Generator) -> pd.DataFrame | pd.Series:
    """Return a copy of ``data`` with every value changed and the index kept.

    Missing values stay missing, dtypes are preserved, and positive columns
    stay positive.
    """
    if isinstance(data, pd.Series):
        return _perturb_series(data, rng)
    out = data.copy()
    for pos in range(out.shape[1]):
        out.isetitem(pos, _perturb_series(data.iloc[:, pos], rng).array)
    return out


def _perturb_series(s: pd.Series, rng: np.random.Generator) -> pd.Series:
    dtype = s.dtype
    n = len(s)
    if n == 0:
        return s.copy()

    if pd.api.types.is_bool_dtype(dtype):
        flip = rng.random(n) < 0.5
        values = s.astype(object).to_numpy()
        flipped = [
            None if pd.isna(v) else (not v if f else v) for v, f in zip(values, flip, strict=True)
        ]
        return pd.Series(flipped, index=s.index, name=s.name).astype(dtype)

    if pd.api.types.is_integer_dtype(dtype):
        missing = s.isna().to_numpy()
        base = s.to_numpy(dtype="float64", na_value=np.nan)
        scaled = np.rint(base * np.exp(rng.normal(0.0, 1.0, n)))
        # Rounding can return the original integer; force a change there.
        bump = rng.choice(np.array([-1.0, 1.0]), size=n)
        if pd.api.types.is_unsigned_integer_dtype(dtype):
            bump = np.ones(n)
        scaled = np.where(scaled == base, base + bump, scaled)
        info = np.iinfo(dtype.numpy_dtype if hasattr(dtype, "numpy_dtype") else dtype)
        scaled = np.clip(scaled, float(info.min), float(info.max))
        out = pd.Series(scaled, index=s.index, name=s.name)
        if missing.any():
            out[missing] = np.nan
        return out.astype(dtype)

    if pd.api.types.is_float_dtype(dtype):
        base = s.to_numpy(dtype="float64", na_value=np.nan)
        finite = base[np.isfinite(base)]
        if finite.size and (finite > 0).all():
            new = base * np.exp(rng.normal(0.0, 1.0, n))
        else:
            scale = float(np.std(finite)) if finite.size else 1.0
            if not np.isfinite(scale) or scale == 0.0:
                scale = 1.0
            new = base + 3.0 * scale * rng.normal(0.0, 1.0, n)
        return pd.Series(new, index=s.index, name=s.name).astype(dtype)

    # Dates, strings, categories and other objects: there is no generic notion
    # of noise, so reorder the existing values instead.
    order = rng.permutation(n)
    shuffled = s.iloc[order]
    shuffled.index = s.index
    return shuffled

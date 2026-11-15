"""A full-sample z-score turns a strategy with no edge into a profitable backtest.

Prices are driftless geometric random walks: tomorrow's return is independent
of everything known today, so no strategy has an edge.  The strategy is plain
mean reversion: short when the price is above its mean, long when below.

* Leaky version: the z-score uses the mean and standard deviation of the whole
  price series.  The mean of a random walk's full path contains its future, so
  "price above the full-sample mean" predicts that later prices are lower.
* Causal version: the same signal with expanding statistics (data up to today).

Both are traded with a one-day lag and scored over the same days.  The leak is
worth a roughly fixed amount of profit per path (about the distance the price
wanders from its mean), so its effect on the Sharpe ratio shrinks as the
backtest gets longer.  The script therefore reports several backtest lengths.
It also runs the auditor on the two signal functions.  Results go to
``results/leak_inflates_backtest.json`` and ``results/leak_inflates_backtest.md``.

Run from the repository root:

    python experiments/leak_inflates_backtest.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import futureleak
from futureleak import audit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

N_PATHS = 1000
LENGTHS = (63, 126, 252, 504, 1260, 2520)  # trading days: a quarter up to ten years
HEADLINE_LENGTH = 252
DAILY_VOL = 0.01
WARMUP = 20  # days before either version is allowed to trade
TRADING_DAYS = 252
AUDIT_SEED = 0


def random_walk(seed: int, n_days: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, DAILY_VOL, n_days)
    index = pd.bdate_range("2010-01-01", periods=n_days)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), index=index, name="close")


def leaky_signal(close: pd.Series) -> pd.Series:
    """Mean reversion on a z-score normalised with full-sample statistics."""
    z = (close - close.mean()) / close.std()
    return -np.sign(z).rename("position")


def causal_signal(close: pd.Series) -> pd.Series:
    """The same signal normalised with statistics of the data seen so far."""
    mean = close.expanding(min_periods=WARMUP).mean()
    std = close.expanding(min_periods=WARMUP).std()
    return -np.sign((close - mean) / std).rename("position")


def sharpe(close: pd.Series, position: pd.Series) -> float:
    """Annualised Sharpe ratio of holding yesterday's position over today's return."""
    returns = close.pct_change()
    pnl = (position.shift(1) * returns).iloc[WARMUP + 1 :]
    return float(pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS))


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)),
        "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
        "min": float(values.min()),
        "max": float(values.max()),
        "share_above_0": float((values > 0.0).mean()),
        "share_above_1": float((values > 1.0).mean()),
        "share_above_2": float((values > 2.0).mean()),
    }


def main() -> int:
    by_length: dict[str, Any] = {}
    for n_days in LENGTHS:
        leaky = np.empty(N_PATHS)
        causal = np.empty(N_PATHS)
        for seed in range(N_PATHS):
            close = random_walk(seed, n_days)
            leaky[seed] = sharpe(close, leaky_signal(close))
            causal[seed] = sharpe(close, causal_signal(close))
        by_length[str(n_days)] = {
            "days": n_days,
            "leaky_sharpe": describe(leaky),
            "causal_sharpe": describe(causal),
        }

    audit_path = random_walk(AUDIT_SEED, HEADLINE_LENGTH)
    leaky_audit = audit(leaky_signal, audit_path, name="leaky_signal")
    causal_audit = audit(causal_signal, audit_path, name="causal_signal")

    payload: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "futureleak": futureleak.__version__,
        },
        "parameters": {
            "paths_per_length": N_PATHS,
            "lengths_days": list(LENGTHS),
            "headline_length_days": HEADLINE_LENGTH,
            "daily_volatility": DAILY_VOL,
            "drift": 0.0,
            "warmup_days": WARMUP,
            "annualisation": TRADING_DAYS,
            "seeds": f"0..{N_PATHS - 1}",
            "audit_path": f"seed {AUDIT_SEED}, {HEADLINE_LENGTH} days",
        },
        "headline": by_length[str(HEADLINE_LENGTH)],
        "by_length": by_length,
        "audit": {
            "leaky_signal": leaky_audit.to_dict(),
            "causal_signal": causal_audit.to_dict(),
        },
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "leak_inflates_backtest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS / "leak_inflates_backtest.md").write_text(_markdown(payload), encoding="utf-8")

    print(f"{N_PATHS} driftless random walks per backtest length; mean annualised Sharpe ratio")
    print("  days   leaky (full-sample z)   causal (expanding z)")
    for entry in by_length.values():
        lk, cs = entry["leaky_sharpe"], entry["causal_sharpe"]
        print(
            f"  {entry['days']:>4}   {lk['mean']:>5.2f} (s.e. {lk['standard_error']:.2f})"
            f"        {cs['mean']:>5.2f} (s.e. {cs['standard_error']:.2f})"
        )
    head = payload["headline"]
    print(
        f"headline ({HEADLINE_LENGTH} days): leaky Sharpe {head['leaky_sharpe']['mean']:.2f}, "
        f"causal Sharpe {head['causal_sharpe']['mean']:.2f}"
    )
    print(f"audit of leaky_signal:  {leaky_audit.verdict} ({leaky_audit.columns[0].kind})")
    print(f"audit of causal_signal: {causal_audit.verdict}")
    print(f"wrote {RESULTS / 'leak_inflates_backtest.json'}")
    print(f"wrote {RESULTS / 'leak_inflates_backtest.md'}")
    return 0


def _markdown(payload: dict[str, Any]) -> str:
    env, params, audits = payload["environment"], payload["parameters"], payload["audit"]
    leaky_col = audits["leaky_signal"]["columns"][0]
    lines = [
        "# A full-sample z-score inflates a backtest with no edge",
        "",
        "Generated by `experiments/leak_inflates_backtest.py`. Do not edit by hand.",
        "",
        f"- Python {env['python']} on {env['platform']}, pandas {env['pandas']}, "
        f"numpy {env['numpy']}, futureleak {env['futureleak']}",
        f"- {params['paths_per_length']} driftless geometric random walks per backtest length "
        f"(seeds {params['seeds']}), daily volatility {params['daily_volatility']}",
        "- Strategy: short when the z-score of price is positive, long when negative, "
        "traded with a one-day lag",
        f"- Both versions are scored on the same days (after a {params['warmup_days']}-day "
        f"warm-up); Sharpe ratios are annualised with {params['annualisation']} days",
        "- leaky: z-score with the mean and standard deviation of the whole path; "
        "causal: expanding mean and standard deviation",
        "",
        "| backtest length (days) | leaky: mean Sharpe | s.e. | paths > 0 | paths > 1 "
        "| paths > 2 | causal: mean Sharpe | s.e. | paths > 0 | paths > 1 | paths > 2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in payload["by_length"].values():
        lk, cs = entry["leaky_sharpe"], entry["causal_sharpe"]
        lines.append(
            f"| {entry['days']} | {lk['mean']:.2f} | {lk['standard_error']:.2f} | "
            f"{lk['share_above_0']:.0%} | {lk['share_above_1']:.0%} | {lk['share_above_2']:.0%} | "
            f"{cs['mean']:.2f} | {cs['standard_error']:.2f} | {cs['share_above_0']:.0%} | "
            f"{cs['share_above_1']:.0%} | {cs['share_above_2']:.0%} |"
        )
    lines += [
        "",
        f"Audit of the two signal functions on one path ({params['audit_path']}):",
        "",
        f"- `leaky_signal`: {audits['leaky_signal']['verdict']}, kind {leaky_col['kind']}, "
        f"{leaky_col['leak_fraction']:.0%} of cut points",
        f"- `causal_signal`: {audits['causal_signal']['verdict']}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())

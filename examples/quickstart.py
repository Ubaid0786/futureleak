"""The example from the README.  Run: python examples/quickstart.py"""

import pandas as pd

from futureleak import audit, synthetic_ohlcv


def features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    return pd.DataFrame(
        {
            "momentum_10": close.pct_change(10),
            "zscore": (close - close.mean()) / close.std(),
            "smooth": close.rolling(5, center=True).mean(),
            "next_return": close.pct_change().shift(-1),
        }
    )


if __name__ == "__main__":
    report = audit(features, synthetic_ohlcv(600, seed=0))
    print(report)
    print()
    print(report.to_markdown())

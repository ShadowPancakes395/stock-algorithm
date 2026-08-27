"""Train and evaluate the volatility forecasting model.

Target: realized volatility (std dev of daily returns) over the NEXT 5
trading days, per stock. This is a regression problem, not classification --
we're forecasting magnitude of movement, not direction.

Why volatility instead of direction: volatility clusters (high-vol periods
tend to follow high-vol periods) in a way next-day/next-5-day direction does
not for large-cap stocks -- confirmed empirically after 7 failed attempts
at direction prediction (see model_classifier_archive.py.bak). This forecast
feeds position sizing and stop-loss width in the rules-based decision layer
(rules.py), not a buy/sell signal directly.

Split: time-based, same reasoning as the archived classifier.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d",
    "price_vs_ma20", "rsi_14", "volatility_20d",
    "volume_ratio", "volume_change_1d",
]
# NOTE: sentiment deliberately excluded -- ablation test showed it
# contributes essentially nothing (R^2 0.1919 without vs 0.1884 with,
# MAE 0.006673 without vs 0.00669 with; without was marginally better on
# both). Also would have required solving a GDELT-tone vs VADER-compound
# scale mismatch for live trading (see live_sentiment.py docstring) for a
# feature that isn't earning its complexity. See chat history for the full
# GDELT sentiment build -- kept as a tested negative result, not deleted.


def build_dataset(featured: pd.DataFrame) -> pd.DataFrame:
    """Add the forward volatility target.

    forward_volatility_5d = std dev of the next 5 daily returns, computed
    by shifting the already-known volatility_20d-style rolling std forward.
    We compute it directly from daily returns to keep the exact 5-day window.
    """
    df = featured.copy()
    daily_return = df.groupby(level="symbol")["close"].pct_change()

    # std of the 5 returns starting the day AFTER today, i.e. a forward-
    # shifted rolling window. rolling() looks backward, so we compute it on
    # the reversed-per-symbol series and shift, then flip back.
    def _forward_vol(g: pd.Series) -> pd.Series:
        return g[::-1].rolling(5).std()[::-1].shift(-1)

    df["forward_volatility_5d"] = (
        daily_return.groupby(level="symbol").transform(_forward_vol)
    )

    cols = FEATURE_COLS + ["forward_volatility_5d"]
    return df.dropna(subset=cols)


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split by timestamp, not randomly."""
    dates = df.index.get_level_values("timestamp").unique().sort_values()
    cutoff = dates[int(len(dates) * (1 - test_frac))]
    train = df[df.index.get_level_values("timestamp") < cutoff]
    test = df[df.index.get_level_values("timestamp") >= cutoff]
    return train, test


def train_model(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_depth=5, learning_rate=0.05, max_iter=300, random_state=42
    )
    model.fit(train[FEATURE_COLS], train["forward_volatility_5d"])
    return model


if __name__ == "__main__":
    from data import fetch_daily_bars
    from features import add_features

    bars = fetch_daily_bars()
    featured = add_features(bars)
    dataset = build_dataset(featured)

    train, test = time_split(dataset)
    print(f"Train rows: {len(train)}, Test rows: {len(test)}")

    model = train_model(train)
    preds = model.predict(test[FEATURE_COLS])
    actual = test["forward_volatility_5d"]

    mae = mean_absolute_error(actual, preds)
    r2 = r2_score(actual, preds)
    # Naive baseline: just use today's trailing 20-day volatility as the
    # forecast. If our model can't beat this, it's not adding value.
    baseline_mae = mean_absolute_error(actual, test["volatility_20d"])

    print(f"Model MAE: {mae:.5f}  (baseline MAE using volatility_20d: {baseline_mae:.5f})")
    print(f"Model R^2: {r2:.4f}")

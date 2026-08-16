"""Train and evaluate the classification model: predict cross-sectional
5-day relative performance.

Target: 1 if a stock's 5-day forward return beats the basket's median
5-day forward return on that date, else 0. This cancels out market-wide
moves (a day where everything is up) and isolates stock-specific signal,
which is what technical indicators can plausibly predict.

Split: time-based (train on earlier dates, test on later) -- never random,
since random splitting would leak future information into training.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d",
    "price_vs_ma20", "rsi_14", "volatility_20d",
    "volume_ratio", "volume_change_1d",
]


def build_dataset(featured: pd.DataFrame) -> pd.DataFrame:
    """Add the cross-sectional target label.

    Order matters: drop rows missing forward_return_5d or features BEFORE
    computing the daily median, otherwise NaN comparisons silently mislabel
    as 0 instead of being excluded.
    """
    df = featured.copy()
    df["forward_return_5d"] = (
        df.groupby(level="symbol")["close"].shift(-5) / df["close"] - 1
    )
    cols = FEATURE_COLS + ["forward_return_5d"]
    df = df.dropna(subset=cols)

    daily_median = df.groupby(level="timestamp")["forward_return_5d"].transform("median")
    df["target"] = (df["forward_return_5d"] > daily_median).astype(int)
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split by timestamp, not randomly -- last test_frac of dates go to test."""
    dates = df.index.get_level_values("timestamp").unique().sort_values()
    cutoff = dates[int(len(dates) * (1 - test_frac))]
    train = df[df.index.get_level_values("timestamp") < cutoff]
    test = df[df.index.get_level_values("timestamp") >= cutoff]
    return train, test


def train_model(train: pd.DataFrame) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42, n_jobs=-1
    )
    model.fit(train[FEATURE_COLS], train["target"])
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

    print(f"Test accuracy: {accuracy_score(test['target'], preds):.4f}")
    print(classification_report(test["target"], preds))

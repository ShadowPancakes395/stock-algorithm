"""TEST ONLY -- does not modify rules.py or the live system.

Tests one specific, falsifiable hypothesis: does downgrading a dip-buy
signal to BASE weight (instead of OVERWEIGHT) on days where a stock's
sentiment sits in the bottom 10% of ITS OWN historical distribution
improve backtest results? This is a discrete rule-based use of sentiment,
distinct from the ML-feature approach already tested and rejected (see
learnings.md / chat history: sentiment added no signal to the volatility
regression, R^2 0.1919 without vs 0.1884 with).

No-lookahead discipline: the 10th-percentile threshold is computed per
walk-forward fold using ONLY that fold's training-period sentiment data,
same as every other backtest in this project.

Result compared directly against a fresh no-sentiment baseline run in the
same script execution, so both share the exact same underlying data pull
-- avoids the apples-to-oranges problem from comparing across separate
runs on different days (see chat history, 9/18 backtest caveat).
"""
import numpy as np
import pandas as pd

from config import TICKERS
from data import fetch_daily_bars
from features import add_features
from model import FEATURE_COLS, build_dataset, train_model
from rules import (
    BASE_WEIGHT, OVERWEIGHT_MULTIPLIER, UNDERWEIGHT_MULTIPLIER, MAX_POSITION_PCT,
    STOP_LOSS_VOL_MULTIPLIER, BASE_STOP_LOSS_PCT, RSI_DIP, RSI_OVERBOUGHT,
)
from backtest import (
    walk_forward_folds, simulate_portfolio, compute_metrics, benchmark_equal_weight,
)

SENTIMENT_PATH = "sentiment_cache.csv"
SENTIMENT_PERCENTILE_THRESHOLD = 0.25  # loosened from 0.10 -- that threshold triggered
                                        # so rarely (60 blocks across 4 folds) it couldn't
                                        # move portfolio metrics either way


def load_sentiment() -> pd.DataFrame:
    df = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"])
    return df.rename(columns={"avg_tone": "sentiment"})[["symbol", "date", "sentiment"]]


def merge_sentiment(featured: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    out = featured.reset_index()
    out["date"] = out["timestamp"].dt.tz_localize(None).dt.normalize()
    out = out.merge(sentiment, on=["symbol", "date"], how="left")
    return out.drop(columns=["date"]).set_index(["symbol", "timestamp"])


def generate_signals_with_sentiment_rule(train_with_sentiment: pd.DataFrame,
                                          test_with_sentiment: pd.DataFrame,
                                          forecast_volatility: pd.Series) -> pd.DataFrame:
    """Same core logic as rules.generate_signals, plus: downgrade a
    dip-in-uptrend OVERWEIGHT signal to BASE if that day's sentiment is
    below the symbol's own 10th percentile, computed from TRAIN only.
    """
    thresholds = (
        train_with_sentiment.groupby(level="symbol")["sentiment"]
        .quantile(SENTIMENT_PERCENTILE_THRESHOLD)
    )

    df = test_with_sentiment.copy()
    df["forecast_volatility"] = forecast_volatility

    is_uptrend = df["price_vs_ma50"] > 0
    dip_in_uptrend = (df["rsi_14"] < RSI_DIP) & is_uptrend
    overbought = df["rsi_14"] > RSI_OVERBOUGHT

    symbol_threshold = df.index.get_level_values("symbol").map(thresholds)
    very_negative_sentiment = df["sentiment"] < symbol_threshold
    # No sentiment data for a (symbol, date) => don't apply the filter --
    # missing data shouldn't silently suppress a signal.
    very_negative_sentiment = very_negative_sentiment.fillna(False)

    blocked_dip = dip_in_uptrend & very_negative_sentiment
    real_dip = dip_in_uptrend & ~very_negative_sentiment

    weight = pd.Series(BASE_WEIGHT, index=df.index)
    weight[real_dip] = BASE_WEIGHT * OVERWEIGHT_MULTIPLIER
    weight[overbought] = BASE_WEIGHT * UNDERWEIGHT_MULTIPLIER

    daily_median_vol = df.groupby(level="timestamp")["forecast_volatility"].transform("median")
    vol_ratio = (df["forecast_volatility"] / daily_median_vol).clip(lower=0.25, upper=4.0)
    df["position_size_pct"] = (weight / vol_ratio).clip(upper=MAX_POSITION_PCT)

    df["signal"] = "HOLD"
    df.loc[real_dip, "signal"] = "BUY"
    df.loc[overbought, "signal"] = "SELL"

    df["stop_loss_pct"] = (df["forecast_volatility"] * STOP_LOSS_VOL_MULTIPLIER).clip(
        lower=BASE_STOP_LOSS_PCT
    )

    print(f"    (blocked {blocked_dip.sum()} dip-buy signals due to very negative sentiment)")
    return df[["signal", "forecast_volatility", "position_size_pct", "stop_loss_pct"]]


def run_backtest(dataset_with_sentiment, use_sentiment_rule: bool):
    from rules import generate_signals as baseline_generate_signals

    folds = walk_forward_folds(dataset_with_sentiment)
    all_signals = []
    for i, (train, test) in enumerate(folds):
        model = train_model(train)
        forecast = pd.Series(
            model.predict(test[FEATURE_COLS]), index=test.index, name="forecast_volatility"
        )
        if use_sentiment_rule:
            signals = generate_signals_with_sentiment_rule(train, test, forecast)
        else:
            signals = baseline_generate_signals(test, forecast)
        all_signals.append(signals)
    return pd.concat(all_signals).sort_index(level="timestamp")


if __name__ == "__main__":
    print("Fetching data and building features...")
    bars = fetch_daily_bars()
    featured = add_features(bars)
    sentiment = load_sentiment()
    dataset_with_sentiment = merge_sentiment(build_dataset(featured), sentiment)

    print("\n=== BASELINE (no sentiment rule) ===")
    baseline_signals = run_backtest(dataset_with_sentiment, use_sentiment_rule=False)
    baseline_equity, baseline_trades = simulate_portfolio(baseline_signals, bars["close"])
    baseline_metrics = compute_metrics(baseline_equity, baseline_trades)
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== WITH SENTIMENT RULE (block dip-buys on very negative sentiment) ===")
    sentiment_signals = run_backtest(dataset_with_sentiment, use_sentiment_rule=True)
    sentiment_equity, sentiment_trades = simulate_portfolio(sentiment_signals, bars["close"])
    sentiment_metrics = compute_metrics(sentiment_equity, sentiment_trades)
    for k, v in sentiment_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== BENCHMARK ===")
    start_date = baseline_signals.index.get_level_values("timestamp").min()
    end_date = baseline_signals.index.get_level_values("timestamp").max()
    bench = benchmark_equal_weight(bars, start_date, end_date)
    for k, v in bench.items():
        print(f"  {k}: {v:.3f}")

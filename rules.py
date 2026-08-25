"""Mostly-invested position-weighting logic, using the volatility model's
forecast for fine-tuning within each band -- not for the core decision.

REDESIGN NOTE: the original version only decided whether to hold a
position at all (rare RSI<30-and-uptrend BUY triggers, cash otherwise).
Backtested result: 7% return vs 97.8% for equal-weight buy-and-hold over
the same ~3yr period -- the strategy was in cash for most of a bull run.
Root cause was structural (under-trading), not bad entry logic (the trades
that did fire had a 61% win rate). This version stays invested by default
across the basket and uses RSI/trend to over/underweight, rather than
gating market participation entirely.

Weighting rules:
  BASE:        default weight when neutral -- equal-weight across the
               basket (1 / len(TICKERS))
  OVERWEIGHT:  RSI < 40 AND price above 50-day MA (buying a dip within an
               uptrend) -- larger-than-base position
  UNDERWEIGHT: RSI > 70 (overbought) -- reduced position, frees capital
  Stop-loss:   still applies per-position, volatility-scaled, as a risk
               backstop independent of the weighting logic above.
"""
import pandas as pd

from config import TICKERS

RSI_DIP = 40
RSI_OVERBOUGHT = 70
BASE_WEIGHT = 1.0 / len(TICKERS)
OVERWEIGHT_MULTIPLIER = 1.5
UNDERWEIGHT_MULTIPLIER = 0.3
MAX_POSITION_PCT = 0.10          # cap per-position size regardless of weighting band
STOP_LOSS_VOL_MULTIPLIER = 3.0   # stop-loss distance = N x forecast daily volatility
BASE_STOP_LOSS_PCT = 0.05        # floor, in case forecast volatility is ~0


def generate_signals(featured: pd.DataFrame, forecast_volatility: pd.Series) -> pd.DataFrame:
    """Given the feature table and a per-row forecast volatility (aligned
    on the same (symbol, timestamp) index), return a target weight
    (BUY/HOLD/SELL relabeled as weight bands) plus stop-loss distance.

    signal column kept as BUY/HOLD/SELL for compatibility with
    backtest.py's trade-log/win-rate accounting (BUY = enter or increase,
    SELL = reduce toward zero, HOLD = maintain).
    """
    df = featured.copy()
    df["forecast_volatility"] = forecast_volatility

    is_uptrend = df["price_vs_ma50"] > 0
    dip_in_uptrend = (df["rsi_14"] < RSI_DIP) & is_uptrend
    overbought = df["rsi_14"] > RSI_OVERBOUGHT

    weight = pd.Series(BASE_WEIGHT, index=df.index)
    weight[dip_in_uptrend] = BASE_WEIGHT * OVERWEIGHT_MULTIPLIER
    weight[overbought] = BASE_WEIGHT * UNDERWEIGHT_MULTIPLIER

    # Volatility still scales sizing within each band, same reasoning as
    # before -- relative to the basket's median forecast volatility that day.
    daily_median_vol = df.groupby(level="timestamp")["forecast_volatility"].transform("median")
    vol_ratio = (df["forecast_volatility"] / daily_median_vol).clip(lower=0.25, upper=4.0)
    df["position_size_pct"] = (weight / vol_ratio).clip(upper=MAX_POSITION_PCT)

    df["signal"] = "HOLD"
    df.loc[dip_in_uptrend, "signal"] = "BUY"
    df.loc[overbought, "signal"] = "SELL"

    df["stop_loss_pct"] = (df["forecast_volatility"] * STOP_LOSS_VOL_MULTIPLIER).clip(
        lower=BASE_STOP_LOSS_PCT
    )

    return df[["signal", "forecast_volatility", "position_size_pct", "stop_loss_pct"]]


if __name__ == "__main__":
    from data import fetch_daily_bars
    from features import add_features, add_sentiment
    from model import FEATURE_COLS, build_dataset, time_split, train_model

    bars = fetch_daily_bars()
    featured = add_features(bars)
    featured = add_sentiment(featured)
    dataset = build_dataset(featured)

    train, test = time_split(dataset)
    model = train_model(train)

    forecast = pd.Series(
        model.predict(test[FEATURE_COLS]), index=test.index, name="forecast_volatility"
    )
    signals = generate_signals(test, forecast)

    print(signals["signal"].value_counts())
    print()
    print(signals["position_size_pct"].describe())

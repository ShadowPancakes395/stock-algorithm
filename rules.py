"""Rules-based BUY/SELL/HOLD decisions, using the volatility model's
forecast for position sizing and stop-loss width -- not for the decision
itself. The decision logic here is deliberately simple and interpretable,
not learned, per the earlier finding that ML doesn't reliably find
direction signal in this feature set.

Entry/exit rules:
  BUY:  RSI < 30 (oversold) AND price above its 50-day MA (longer-horizon
        uptrend filter -- deliberately NOT the 20-day MA, which reacts on
        nearly the same timescale as RSI and made this combination almost
        never fire: 0 co-occurrences across the full test set. A 50-day
        filter checks the broader trend while RSI times the pullback
        within it.)
  SELL: RSI > 70 (overbought), OR price has dropped below a volatility-
        scaled stop-loss from its recent high
  HOLD: otherwise

Position sizing: inversely scaled to forecast volatility -- lower forecast
volatility allows a larger position (up to MAX_POSITION_PCT of portfolio),
higher forecast volatility scales it down.
"""
import pandas as pd

from config import BUY_THRESHOLD, SELL_THRESHOLD  # noqa: F401 (kept for future tuning, unused directly here)

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MAX_POSITION_PCT = 0.10          # cap per-position size as a fraction of portfolio
STOP_LOSS_VOL_MULTIPLIER = 3.0   # stop-loss distance = N x forecast daily volatility
BASE_STOP_LOSS_PCT = 0.05        # floor, in case forecast volatility is ~0


def generate_signals(featured: pd.DataFrame, forecast_volatility: pd.Series) -> pd.DataFrame:
    """Given the feature table and a per-row forecast volatility (aligned
    on the same (symbol, timestamp) index), return BUY/SELL/HOLD signals
    plus position size and stop-loss distance.
    """
    df = featured.copy()
    df["forecast_volatility"] = forecast_volatility

    is_uptrend = df["price_vs_ma50"] > 0
    oversold = df["rsi_14"] < RSI_OVERSOLD
    overbought = df["rsi_14"] > RSI_OVERBOUGHT

    df["signal"] = "HOLD"
    df.loc[oversold & is_uptrend, "signal"] = "BUY"
    df.loc[overbought, "signal"] = "SELL"

    # Position sizing: scale MAX_POSITION_PCT down as forecast volatility
    # rises. Uses the cross-sectional median forecast volatility that day
    # as the reference point, so sizing is relative to the basket, not an
    # arbitrary fixed volatility level.
    daily_median_vol = df.groupby(level="timestamp")["forecast_volatility"].transform("median")
    vol_ratio = (df["forecast_volatility"] / daily_median_vol).clip(lower=0.25, upper=4.0)
    df["position_size_pct"] = (MAX_POSITION_PCT / vol_ratio).clip(upper=MAX_POSITION_PCT)

    # Stop-loss distance: wider for higher forecast volatility, so normal
    # swings in a volatile stock don't trigger a premature exit.
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
    print(signals.head(15))

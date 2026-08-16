"""Compute technical indicator features from daily OHLCV bars.

Input: DataFrame indexed by (symbol, timestamp) with open/high/low/close/volume,
as returned by data.fetch_daily_bars().

Output: same index, with feature columns added. Operates per-symbol so
indicators never leak across tickers.
"""
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _features_for_one_symbol(g: pd.DataFrame) -> pd.DataFrame:
    close = g["close"]

    g["return_1d"] = close.pct_change(1)
    g["return_5d"] = close.pct_change(5)
    g["return_20d"] = close.pct_change(20)

    g["ma_20"] = close.rolling(20).mean()
    g["price_vs_ma20"] = close / g["ma_20"] - 1

    g["rsi_14"] = _rsi(close, 14)

    g["volatility_20d"] = close.pct_change().rolling(20).std()

    volume = g["volume"]
    volume_ma20 = volume.rolling(20).mean()
    g["volume_ratio"] = volume / volume_ma20
    g["volume_change_1d"] = volume.pct_change(1)

    return g


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator columns, computed independently per symbol."""
    out = df.groupby(level="symbol", group_keys=False).apply(_features_for_one_symbol)
    return out


if __name__ == "__main__":
    from data import fetch_daily_bars

    bars = fetch_daily_bars()
    featured = add_features(bars)
    print(featured.shape)
    print(featured.dropna().head())

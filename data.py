"""Pull daily bar data from Alpaca for the configured ticker basket."""
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, TICKERS

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def fetch_daily_bars(tickers: list[str] = TICKERS, years: int = 5) -> pd.DataFrame:
    """Fetch daily OHLCV bars for the given tickers over the trailing N years.

    Returns a DataFrame indexed by (symbol, timestamp) with columns:
    open, high, low, close, volume.
    """
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,  # free-tier accounts don't have SIP access
    )
    bars = client.get_stock_bars(request)
    df = bars.df  # MultiIndex (symbol, timestamp)
    return df[["open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    df = fetch_daily_bars()
    print(df.shape)
    print(df.head())

"""Daily performance tracker: logs the paper account's actual value
alongside what an equal-weight buy-and-hold benchmark on the same 28
tickers would be worth, starting from the same date/capital. Appends one
row per day to performance_log.csv -- run this once per day, same
schedule as trade.py (after it, so today's trades are reflected).

This is the actual answer to "are these good trades" -- not any single
day's signals in isolation, but the cumulative strategy-vs-benchmark
comparison over time, same methodology as backtest.py's benchmark check.
"""
import os
from datetime import date

import pandas as pd
from alpaca.trading.client import TradingClient

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER, TICKERS
from data import fetch_daily_bars

LOG_PATH = "performance_log.csv"
START_DATE = "2026-08-26"  # first day of live paper trading -- benchmark starts here too
START_CAPITAL = 100_000


def get_benchmark_value(bars: pd.DataFrame) -> float:
    """Equal-weight buy-and-hold value on the same 28 tickers, same start
    date/capital as the paper account, valued as of the most recent
    available bar (today's, if the market's open and data has landed).
    """
    prices = bars["close"].unstack(level="symbol")
    prices = prices[prices.index >= START_DATE]
    normalized = prices / prices.iloc[0]
    equal_weight_return = normalized.iloc[-1].mean()
    return START_CAPITAL * equal_weight_return


if __name__ == "__main__":
    assert ALPACA_PAPER, "Refusing to run against a non-paper account."

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    account = client.get_account()
    strategy_value = float(account.portfolio_value)

    bars = fetch_daily_bars(TICKERS, years=1)
    benchmark_value = get_benchmark_value(bars)

    row = pd.DataFrame([{
        "date": date.today().isoformat(),
        "strategy_value": strategy_value,
        "benchmark_value": benchmark_value,
        "strategy_return_pct": (strategy_value / START_CAPITAL - 1) * 100,
        "benchmark_return_pct": (benchmark_value / START_CAPITAL - 1) * 100,
    }])

    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
        # Overwrite today's row if this is a re-run, rather than duplicate.
        existing = existing[existing["date"] != row.iloc[0]["date"]]
        row = pd.concat([existing, row], ignore_index=True)

    row.to_csv(LOG_PATH, index=False)
    print(row.to_string(index=False))

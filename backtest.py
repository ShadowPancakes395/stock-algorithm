"""Walk-forward backtest of the combined rules + volatility-sizing system.

This is the actual validation gate before paper trading: simulates real
portfolio mechanics (cash, positions, entry/exit) using out-of-sample
signals only, and compares against equal-weight buy-and-hold as a
benchmark. A model/rules combo that merely reports OK-looking component
metrics (like model.py's R^2 or rules.py's signal counts) hasn't proven
anything about actual trading performance -- this has to.

Walk-forward: expanding training window, 4 folds. Each fold retrains the
volatility model on all data up to that point, then generates signals for
the next chunk. This produces one continuous out-of-sample equity curve
across ~60% of the full date range, rather than a single train/test split.
"""
import numpy as np
import pandas as pd

from config import TICKERS
from data import fetch_daily_bars
from features import add_features, add_sentiment
from model import FEATURE_COLS, build_dataset, train_model
from rules import generate_signals

INITIAL_CAPITAL = 100_000
N_FOLDS = 4
MAX_OPEN_POSITIONS = 10  # cap total concurrent positions, independent of per-position sizing


def walk_forward_folds(df: pd.DataFrame, n_folds: int = N_FOLDS):
    """Expanding-window folds: first ~40% of dates as the initial training
    period, remaining ~60% split into n_folds sequential test chunks, each
    fold's training data expanding to include all prior folds' test data.
    """
    dates = df.index.get_level_values("timestamp").unique().sort_values()
    initial_train_end = dates[int(len(dates) * 0.4)]
    remaining = dates[dates > initial_train_end]
    fold_bounds = np.array_split(remaining, n_folds)

    folds = []
    train_end = initial_train_end
    for fold_dates in fold_bounds:
        test_start, test_end = fold_dates[0], fold_dates[-1]
        train = df[df.index.get_level_values("timestamp") <= train_end]
        test = df[
            (df.index.get_level_values("timestamp") >= test_start)
            & (df.index.get_level_values("timestamp") <= test_end)
        ]
        folds.append((train, test))
        train_end = test_end
    return folds


def generate_all_signals(dataset: pd.DataFrame) -> pd.DataFrame:
    """Run the walk-forward loop, returning combined out-of-sample signals
    (BUY/SELL/HOLD + sizing) for every fold's test period, concatenated.
    """
    folds = walk_forward_folds(dataset)
    all_signals = []

    for i, (train, test) in enumerate(folds):
        model = train_model(train)
        forecast = pd.Series(
            model.predict(test[FEATURE_COLS]), index=test.index, name="forecast_volatility"
        )
        signals = generate_signals(test, forecast)
        print(f"  Fold {i+1}: trained on {len(train)} rows, "
              f"signals for {len(test)} rows "
              f"({(signals['signal'] == 'BUY').sum()} BUY, "
              f"{(signals['signal'] == 'SELL').sum()} SELL)")
        all_signals.append(signals)

    return pd.concat(all_signals).sort_index(level="timestamp")


def simulate_portfolio(signals: pd.DataFrame, close_prices: pd.Series) -> pd.DataFrame:
    """Simulate trading the signals day by day. Returns a DataFrame of
    portfolio value indexed by date (the equity curve).

    Mechanics: mostly-invested by default (see rules.py). Each symbol has
    a target weight band each day (BASE/OVERWEIGHT/UNDERWEIGHT, encoded as
    HOLD/BUY/SELL in the signal column). When a held position's band
    changes, the position is closed and reopened at the new target size
    -- a deliberate simplification to avoid weighted-average cost-basis
    tracking across partial rebalances. Stop-losses are checked before
    that day's band logic, so risk management takes priority.
    """
    cash = INITIAL_CAPITAL
    positions = {}  # symbol -> dict(shares, entry_price, stop_loss_price, band)
    equity_curve = []
    trade_log = []

    dates = signals.index.get_level_values("timestamp").unique().sort_values()
    for date in dates:
        day = signals.xs(date, level="timestamp")
        prices_today = close_prices.xs(date, level="timestamp")

        def _close_position(symbol, price):
            nonlocal cash
            cash += positions[symbol]["shares"] * price
            trade_log.append(price / positions[symbol]["entry_price"] - 1)
            del positions[symbol]

        # 1. Check stop-losses on existing positions first.
        for symbol in list(positions.keys()):
            if symbol not in prices_today.index:
                continue
            price = prices_today[symbol]
            if price <= positions[symbol]["stop_loss_price"]:
                _close_position(symbol, price)

        # 2. Close positions whose weight band changed (BASE/OVER/UNDER).
        for symbol in list(positions.keys()):
            if symbol not in day.index:
                continue
            price = prices_today.get(symbol)
            if price is None:
                continue
            desired_band = {"BUY": "OVER", "SELL": "UNDER", "HOLD": "BASE"}[day.loc[symbol, "signal"]]
            if desired_band != positions[symbol]["band"]:
                _close_position(symbol, price)

        # 3. Open/reopen positions at their current target size, respecting
        # the open-position cap. Never proactively enters an UNDERWEIGHT
        # (overbought) name from flat -- only BASE or OVERWEIGHT.
        portfolio_value = cash + sum(
            positions[s]["shares"] * prices_today.get(s, positions[s]["entry_price"])
            for s in positions
        )
        for symbol in day.index:
            if symbol in positions:
                continue
            if len(positions) >= MAX_OPEN_POSITIONS:
                break
            signal = day.loc[symbol, "signal"]
            if signal == "SELL":  # underweight/overbought -- don't enter fresh
                continue
            price = prices_today.get(symbol)
            if price is None or price <= 0:
                continue
            allocation = portfolio_value * day.loc[symbol, "position_size_pct"]
            if allocation > cash or allocation <= 0:
                continue
            shares = allocation / price
            stop_loss_price = price * (1 - day.loc[symbol, "stop_loss_pct"])
            band = "OVER" if signal == "BUY" else "BASE"
            positions[symbol] = {
                "shares": shares, "entry_price": price,
                "stop_loss_price": stop_loss_price, "band": band,
            }
            cash -= allocation

        total_value = cash + sum(
            positions[s]["shares"] * prices_today.get(s, positions[s]["entry_price"])
            for s in positions
        )
        equity_curve.append({"date": date, "portfolio_value": total_value, "open_positions": len(positions)})

    return pd.DataFrame(equity_curve).set_index("date"), trade_log


def compute_metrics(equity_curve: pd.DataFrame, trade_log: list) -> dict:
    values = equity_curve["portfolio_value"]
    daily_returns = values.pct_change().dropna()

    total_return = values.iloc[-1] / values.iloc[0] - 1
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    max_drawdown = drawdown.min()
    win_rate = np.mean([t > 0 for t in trade_log]) if trade_log else float("nan")

    return {
        "total_return_pct": total_return * 100,
        "annualized_sharpe": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
        "num_trades": len(trade_log),
        "win_rate_pct": win_rate * 100 if trade_log else float("nan"),
    }


def benchmark_equal_weight(bars: pd.DataFrame, start_date, end_date) -> dict:
    """Equal-weight buy-and-hold across all tickers over the same period,
    for a fair comparison against the strategy's out-of-sample results.
    """
    prices = bars["close"].unstack(level="symbol")
    prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]
    normalized = prices / prices.iloc[0]
    portfolio = normalized.mean(axis=1) * INITIAL_CAPITAL

    daily_returns = portfolio.pct_change().dropna()
    total_return = portfolio.iloc[-1] / portfolio.iloc[0] - 1
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    running_max = portfolio.cummax()
    max_drawdown = ((portfolio - running_max) / running_max).min()

    return {
        "total_return_pct": total_return * 100,
        "annualized_sharpe": sharpe,
        "max_drawdown_pct": max_drawdown * 100,
    }


if __name__ == "__main__":
    print("Fetching data and building features...")
    bars = fetch_daily_bars()
    featured = add_features(bars)
    featured = add_sentiment(featured)
    dataset = build_dataset(featured)

    print("\nRunning walk-forward folds...")
    signals = generate_all_signals(dataset)

    print("\nSimulating portfolio...")
    close_prices = bars["close"]
    equity_curve, trade_log = simulate_portfolio(signals, close_prices)

    print("\n=== STRATEGY RESULTS (out-of-sample) ===")
    metrics = compute_metrics(equity_curve, trade_log)
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== BENCHMARK: equal-weight buy-and-hold, same period ===")
    start_date = signals.index.get_level_values("timestamp").min()
    end_date = signals.index.get_level_values("timestamp").max()
    bench = benchmark_equal_weight(bars, start_date, end_date)
    for k, v in bench.items():
        print(f"  {k}: {v:.3f}")

    equity_curve.to_csv("backtest_equity_curve.csv")

"""Research-only monthly momentum backtest, isolated from live trading."""
import numpy as np
import pandas as pd
from pathlib import Path

from config import TICKERS

MOMENTUM_LOOKBACK_SESSIONS = 252
MOMENTUM_SKIP_SESSIONS = 21
TOP_N = 10
INITIAL_CAPITAL = 100_000.0
SLIPPAGE_BPS = 5.0
TRADING_SESSIONS_PER_YEAR = 252


def _price_matrices(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete open/close matrices in configured ticker order."""
    if not isinstance(bars.index, pd.MultiIndex) or not {
        "symbol", "timestamp"
    }.issubset(bars.index.names):
        raise ValueError("bars must use a (symbol, timestamp) MultiIndex")
    if bars.index.has_duplicates:
        raise ValueError("bars contain duplicate (symbol, timestamp) rows")
    missing_columns = {"open", "close"} - set(bars.columns)
    if missing_columns:
        raise ValueError(f"bars are missing required columns: {sorted(missing_columns)}")

    dates = pd.DatetimeIndex(
        bars.index.get_level_values("timestamp").unique()
    ).sort_values()
    closes = bars["close"].unstack("symbol").reindex(
        index=dates, columns=TICKERS
    )
    opens = bars["open"].unstack("symbol").reindex(
        index=dates, columns=TICKERS
    )

    invalid_close = ~np.isfinite(closes.to_numpy(dtype=float)) | (
        closes.to_numpy(dtype=float) <= 0
    )
    if invalid_close.any():
        row, column = np.argwhere(invalid_close)[0]
        raise ValueError(
            f"missing or invalid close for {TICKERS[column]} on {dates[row].date()}"
        )
    return opens, closes


def _month_end_positions(dates: pd.DatetimeIndex) -> np.ndarray:
    month_dates = dates.tz_localize(None) if dates.tz is not None else dates
    month_keys = month_dates.to_period("M")
    positions = pd.Series(np.arange(len(dates)), index=month_keys)
    return positions.groupby(level=0).last().to_numpy(dtype=int)


def _validate_execution_opens(
    opens: pd.DataFrame, execution_date: pd.Timestamp
) -> None:
    values = opens.loc[execution_date].to_numpy(dtype=float)
    invalid = ~np.isfinite(values) | (values <= 0)
    if invalid.any():
        column = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            f"missing or invalid open for {TICKERS[column]} on {execution_date.date()}"
        )


def monthly_momentum_targets(bars: pd.DataFrame) -> pd.DataFrame:
    """Select the top 10 tickers monthly by 12-1 momentum.

    Scores use the adjusted close 21 sessions before signal date divided by
    the adjusted close 252 sessions before signal date, minus one. Targets
    are indexed by the following session's execution date and ticker.
    """
    opens, closes = _price_matrices(bars)
    dates = closes.index
    records = []

    for signal_position in _month_end_positions(dates):
        if signal_position < MOMENTUM_LOOKBACK_SESSIONS:
            continue
        execution_position = signal_position + 1
        if execution_position >= len(dates):
            continue

        signal_date = dates[signal_position]
        execution_date = dates[execution_position]
        _validate_execution_opens(opens, execution_date)

        start_prices = closes.iloc[
            signal_position - MOMENTUM_LOOKBACK_SESSIONS
        ]
        skip_prices = closes.iloc[
            signal_position - MOMENTUM_SKIP_SESSIONS
        ]
        scores = skip_prices / start_prices - 1.0
        ranked = pd.DataFrame(
            {"symbol": TICKERS, "score": scores.reindex(TICKERS).to_numpy()}
        ).sort_values(
            ["score", "symbol"], ascending=[False, True], kind="mergesort"
        )

        for row in ranked.head(TOP_N).itertuples(index=False):
            records.append(
                {
                    "execution_date": execution_date,
                    "symbol": row.symbol,
                    "signal_date": signal_date,
                    "score": float(row.score),
                    "target_weight": 1.0 / TOP_N,
                }
            )

    columns = ["signal_date", "score", "target_weight"]
    if not records:
        empty_index = pd.MultiIndex.from_arrays(
            [[], []], names=["execution_date", "symbol"]
        )
        return pd.DataFrame(columns=columns, index=empty_index)
    return pd.DataFrame(records).set_index(["execution_date", "symbol"])[columns]


def calculate_metrics(
    equity: pd.Series, initial_capital: float = INITIAL_CAPITAL
) -> dict:
    """Calculate returns, annualized Sharpe, and drawdown from daily equity."""
    if initial_capital <= 0 or not np.isfinite(initial_capital):
        raise ValueError("initial_capital must be finite and positive")
    values = equity.astype(float)
    if values.empty or not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
        raise ValueError("equity must contain finite positive values")
    if not values.index.is_monotonic_increasing or values.index.has_duplicates:
        raise ValueError("equity index must be unique and sorted chronologically")

    daily_returns = values.pct_change().dropna().to_numpy()
    daily_returns = np.insert(daily_returns, 0, values.iloc[0] / initial_capital - 1.0)
    daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    sharpe = (
        float(np.mean(daily_returns) / daily_std * np.sqrt(TRADING_SESSIONS_PER_YEAR))
        if daily_std > 0
        else 0.0
    )

    equity_with_initial = np.insert(values.to_numpy(), 0, initial_capital)
    running_peak = np.maximum.accumulate(equity_with_initial)
    drawdowns = equity_with_initial / running_peak - 1.0
    elapsed_sessions = len(values)
    total_return = values.iloc[-1] / initial_capital - 1.0
    annualized_return = (values.iloc[-1] / initial_capital) ** (
        TRADING_SESSIONS_PER_YEAR / elapsed_sessions
    ) - 1.0
    return {
        "total_return_pct": float(total_return * 100.0),
        "annualized_return_pct": float(annualized_return * 100.0),
        "annualized_sharpe": sharpe,
        "max_drawdown_pct": float(drawdowns.min() * 100.0),
    }


def _validate_targets(targets: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    if not isinstance(targets.index, pd.MultiIndex) or not {
        "execution_date", "symbol"
    }.issubset(targets.index.names):
        raise ValueError("targets must use an (execution_date, symbol) MultiIndex")
    if targets.index.has_duplicates:
        raise ValueError("targets contain duplicate execution_date/symbol rows")
    if "target_weight" not in targets.columns:
        raise ValueError("targets are missing target_weight")
    if targets.empty:
        raise ValueError("targets contain no executable dates")

    frame = targets.reset_index().copy()
    if not frame["symbol"].isin(TICKERS).all():
        invalid_symbol = frame.loc[~frame["symbol"].isin(TICKERS), "symbol"].iloc[0]
        raise ValueError(f"target contains unknown symbol {invalid_symbol}")
    unknown_dates = ~frame["execution_date"].isin(dates)
    if unknown_dates.any():
        invalid_date = frame.loc[unknown_dates, "execution_date"].iloc[0]
        raise ValueError(f"target execution date {invalid_date} is not in price data")

    for execution_date, rows in frame.groupby("execution_date", sort=True):
        weights = rows["target_weight"].to_numpy(dtype=float)
        if len(rows) != TOP_N or not np.isfinite(weights).all():
            raise ValueError(f"invalid target set on {execution_date.date()}: expected {TOP_N} weights")
        if (weights <= 0).any() or not np.allclose(weights, 1.0 / TOP_N):
            raise ValueError(f"invalid target weights on {execution_date.date()}")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"target weights do not sum to 1 on {execution_date.date()}")
    return frame


def simulate_portfolio(
    bars: pd.DataFrame,
    targets: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    slippage_bps: float = SLIPPAGE_BPS,
) -> tuple[pd.Series, dict]:
    """Simulate monthly target rebalances at next-session adjusted opens."""
    if initial_capital <= 0 or not np.isfinite(initial_capital):
        raise ValueError("initial_capital must be finite and positive")
    if slippage_bps < 0 or not np.isfinite(slippage_bps):
        raise ValueError("slippage_bps must be finite and nonnegative")
    opens, closes = _price_matrices(bars)
    target_rows = _validate_targets(targets, closes.index)
    execution_dates = pd.DatetimeIndex(
        target_rows["execution_date"].drop_duplicates().sort_values()
    )
    targets_by_date = {
        execution_date: rows.set_index("symbol")["target_weight"]
        for execution_date, rows in target_rows.groupby("execution_date", sort=True)
    }
    for execution_date in execution_dates:
        _validate_execution_opens(opens, execution_date)

    rate = slippage_bps / 10_000.0
    cash = float(initial_capital)
    positions: dict[str, float] = {}
    total_turnover = 0.0
    total_slippage = 0.0
    equity_values = []
    evaluation_dates = closes.index[closes.index >= execution_dates[0]]

    for session_date in evaluation_dates:
        if session_date in targets_by_date:
            session_opens = opens.loc[session_date]
            pretrade_equity = cash + sum(
                quantity * float(session_opens[symbol])
                for symbol, quantity in positions.items()
            )
            target_weights = targets_by_date[session_date]
            traded_symbols = set(positions) | set(target_weights.index)
            target_equity = pretrade_equity

            for _ in range(20):
                target_quantities = {
                    symbol: (
                        float(target_weights[symbol]) * target_equity
                        / float(session_opens[symbol])
                        if symbol in target_weights
                        else 0.0
                    )
                    for symbol in traded_symbols
                }
                estimated_slippage = sum(
                    abs(target_quantities[symbol] - positions.get(symbol, 0.0))
                    * float(session_opens[symbol])
                    * rate
                    for symbol in traded_symbols
                )
                updated_target_equity = pretrade_equity - estimated_slippage
                if abs(updated_target_equity - target_equity) < 1e-9:
                    target_equity = updated_target_equity
                    break
                target_equity = updated_target_equity
            else:
                raise RuntimeError(f"slippage sizing did not converge on {session_date.date()}")

            target_quantities = {
                symbol: (
                    float(target_weights[symbol]) * target_equity
                    / float(session_opens[symbol])
                    if symbol in target_weights
                    else 0.0
                )
                for symbol in traded_symbols
            }
            for symbol in traded_symbols:
                quantity_change = target_quantities[symbol] - positions.get(symbol, 0.0)
                trade_notional = abs(quantity_change) * float(session_opens[symbol])
                slippage = trade_notional * rate
                cash -= quantity_change * float(session_opens[symbol]) + slippage
                total_turnover += trade_notional / pretrade_equity
                total_slippage += slippage
                if target_quantities[symbol] > 0:
                    positions[symbol] = target_quantities[symbol]
                else:
                    positions.pop(symbol, None)

            if cash < -1e-6:
                raise RuntimeError(f"rebalance created negative cash on {session_date.date()}")
            if abs(cash) < 1e-8:
                cash = 0.0

        session_closes = closes.loc[session_date]
        equity_value = cash + sum(
            quantity * float(session_closes[symbol])
            for symbol, quantity in positions.items()
        )
        equity_values.append(equity_value)

    equity = pd.Series(equity_values, index=evaluation_dates, name="candidate_equity")
    summary = calculate_metrics(equity, initial_capital=initial_capital)
    summary.update(
        {
            "rebalance_count": int(len(execution_dates)),
            "gross_turnover": float(total_turnover),
            "slippage_dollars": float(total_slippage),
        }
    )
    return equity, summary


def simulate_equal_weight_benchmark(
    bars: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float = INITIAL_CAPITAL,
    slippage_bps: float = SLIPPAGE_BPS,
) -> pd.Series:
    """Buy all configured tickers equally at the start open, then hold."""
    if initial_capital <= 0 or not np.isfinite(initial_capital):
        raise ValueError("initial_capital must be finite and positive")
    if slippage_bps < 0 or not np.isfinite(slippage_bps):
        raise ValueError("slippage_bps must be finite and nonnegative")
    opens, closes = _price_matrices(bars)
    if start_date not in closes.index or end_date not in closes.index:
        raise ValueError("benchmark start and end dates must be present in price data")
    if start_date > end_date:
        raise ValueError("benchmark start_date must not follow end_date")
    _validate_execution_opens(opens, start_date)

    rate = slippage_bps / 10_000.0
    invested_value = initial_capital / (1.0 + rate)
    per_ticker_value = invested_value / len(TICKERS)
    quantities = {
        symbol: per_ticker_value / float(opens.loc[start_date, symbol])
        for symbol in TICKERS
    }
    dates = closes.index[(closes.index >= start_date) & (closes.index <= end_date)]
    values = [
        sum(quantity * float(closes.loc[date, symbol]) for symbol, quantity in quantities.items())
        for date in dates
    ]
    return pd.Series(values, index=dates, name="benchmark_equity")


def run_backtest(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run candidate and benchmark over identical valuation dates."""
    targets = monthly_momentum_targets(bars)
    if targets.empty:
        raise ValueError("not enough historical sessions to create a momentum target")

    execution_dates = targets.index.get_level_values("execution_date")
    start_date = execution_dates.min()
    _, closes = _price_matrices(bars)
    end_date = closes.index[-1]

    candidate_equity, candidate_results = simulate_portfolio(bars, targets)
    benchmark_equity = simulate_equal_weight_benchmark(
        bars, start_date=start_date, end_date=end_date
    )
    curves = pd.concat([candidate_equity, benchmark_equity], axis=1)
    if curves.isna().any().any():
        raise ValueError("candidate and benchmark equity dates do not align")

    benchmark_results = calculate_metrics(benchmark_equity)
    benchmark_results.update(
        {
            "purchase_count": 1,
            "gross_turnover": 1.0 / (1.0 + SLIPPAGE_BPS / 10_000.0),
            "slippage_dollars": INITIAL_CAPITAL
            - INITIAL_CAPITAL / (1.0 + SLIPPAGE_BPS / 10_000.0),
        }
    )
    results = {
        "candidate": candidate_results,
        "benchmark": benchmark_results,
        "excess_cumulative_return_pct": (
            candidate_results["total_return_pct"]
            - benchmark_results["total_return_pct"]
        ),
        "excess_annualized_return_pct": (
            candidate_results["annualized_return_pct"]
            - benchmark_results["annualized_return_pct"]
        ),
        "signal_start_date": targets["signal_date"].min(),
        "execution_start_date": start_date,
        "end_date": end_date,
    }
    return curves, results


def _print_portfolio_report(name: str, results: dict) -> None:
    print(
        f"{name}: cumulative {results['total_return_pct']:.2f}% | "
        f"annualized {results['annualized_return_pct']:.2f}% | "
        f"max drawdown {results['max_drawdown_pct']:.2f}% | "
        f"Sharpe {results['annualized_sharpe']:.2f}"
    )
    if "rebalance_count" in results:
        print(
            f"  Rebalance dates: {results['rebalance_count']} | "
            f"cumulative gross turnover: {results['gross_turnover']:.2f}x | "
            f"estimated slippage: ${results['slippage_dollars']:.2f}"
        )
    else:
        print(
            f"  Initial buy-and-hold purchase | "
            f"estimated slippage: ${results['slippage_dollars']:.2f}"
        )


def main() -> None:
    """Fetch historical bars and write a local research report."""
    from data import fetch_daily_bars

    bars = fetch_daily_bars(years=5)
    curves, results = run_backtest(bars)

    print("FORMULA MOMENTUM RESEARCH BACKTEST — no orders submitted")
    print(
        f"Signal period: {results['signal_start_date'].date()} onward; "
        f"execution: next session open; valuation through {results['end_date'].date()}"
    )
    print(
        f"Rule: {MOMENTUM_LOOKBACK_SESSIONS}-session return skipping "
        f"{MOMENTUM_SKIP_SESSIONS} sessions; top {TOP_N}; monthly; "
        f"slippage {SLIPPAGE_BPS:.1f} bps per side"
    )
    _print_portfolio_report("Candidate", results["candidate"])
    _print_portfolio_report("Equal-weight buy-and-hold benchmark", results["benchmark"])
    print(
        f"Excess return: {results['excess_cumulative_return_pct']:.2f} percentage "
        f"points cumulative; {results['excess_annualized_return_pct']:.2f} "
        "percentage points annualized"
    )
    print(
        "Limitation: this uses the currently configured 28-stock basket for all "
        "historical dates, so survivorship and universe-selection effects may bias results."
    )

    output_path = Path(__file__).with_name("momentum_backtest_equity_curve.csv")
    curves.to_csv(output_path, index_label="date")
    print(f"Equity curve saved to {output_path}")


if __name__ == "__main__":
    main()

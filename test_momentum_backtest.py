import unittest

import numpy as np
import pandas as pd

from config import TICKERS
from momentum_backtest import (
    calculate_metrics,
    monthly_momentum_targets,
    run_backtest,
    simulate_equal_weight_benchmark,
    simulate_portfolio,
)


def make_bars(session_count: int = 620) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=session_count)
    index = pd.MultiIndex.from_product(
        [TICKERS, dates], names=["symbol", "timestamp"]
    )
    bars = pd.DataFrame(index=index)
    bars["open"] = 100.0
    bars["close"] = 100.0
    bars["high"] = 100.0
    bars["low"] = 100.0
    bars["volume"] = 1_000_000
    return bars.sort_index()


def month_end_positions(dates: pd.DatetimeIndex) -> list[int]:
    month_keys = dates.to_period("M")
    positions = pd.Series(np.arange(len(dates)), index=month_keys)
    return positions.groupby(level=0).last().astype(int).tolist()


def make_targets(execution_dates, selections) -> pd.DataFrame:
    records = []
    for execution_date, symbols in zip(execution_dates, selections):
        for symbol in symbols:
            records.append(
                {
                    "execution_date": execution_date,
                    "symbol": symbol,
                    "signal_date": execution_date - pd.offsets.BDay(1),
                    "score": 1.0,
                    "target_weight": 0.10,
                }
            )
    return pd.DataFrame(records).set_index(["execution_date", "symbol"])


class MonthlyMomentumTargetsTests(unittest.TestCase):
    def test_ranks_using_12_month_return_skipping_latest_month(self):
        bars = make_bars()
        dates = bars.index.get_level_values("timestamp").unique().sort_values()
        signal_position = next(
            position
            for position in reversed(month_end_positions(dates))
            if position >= 252 and position + 1 < len(dates)
        )
        lookback_position = signal_position - 252
        skip_position = signal_position - 21

        for rank, symbol in enumerate(TICKERS):
            bars.loc[(symbol, dates[lookback_position]), "close"] = 100.0
            bars.loc[(symbol, dates[skip_position]), "close"] = (
                150.0 + rank if rank < 10 else 100.0
            )

        targets = monthly_momentum_targets(bars)
        execution_date = dates[signal_position + 1]
        selected = targets.xs(execution_date, level="execution_date")

        expected_symbols = sorted(TICKERS[:10], key=lambda symbol: TICKERS.index(symbol), reverse=True)
        self.assertEqual(list(selected.index), expected_symbols)
        self.assertAlmostEqual(selected.loc[TICKERS[9], "score"], 0.59)
        self.assertEqual(selected["signal_date"].unique().tolist(), [dates[signal_position]])

    def test_targets_are_month_end_signals_executed_next_session_at_ten_percent(self):
        bars = make_bars()
        dates = bars.index.get_level_values("timestamp").unique().sort_values()
        targets = monthly_momentum_targets(bars)

        for execution_date, selected in targets.groupby(level="execution_date", sort=True):
            execution_position = dates.get_loc(execution_date)
            signal_date = selected["signal_date"].iloc[0]
            signal_position = dates.get_loc(signal_date)
            same_month_positions = [
                position
                for position in range(len(dates))
                if dates[position].to_period("M") == signal_date.to_period("M")
            ]
            self.assertEqual(signal_position, same_month_positions[-1])
            self.assertEqual(execution_position, signal_position + 1)
            self.assertEqual(len(selected), 10)
            np.testing.assert_allclose(selected["target_weight"].to_numpy(), 0.10)
            self.assertAlmostEqual(selected["target_weight"].sum(), 1.0)

    def test_no_target_is_emitted_without_full_lookback_and_next_session(self):
        bars = make_bars(session_count=252)

        targets = monthly_momentum_targets(bars)

        self.assertTrue(targets.empty)
        self.assertEqual(
            list(targets.columns), ["signal_date", "score", "target_weight"]
        )

    def test_missing_close_fails_with_symbol_and_date(self):
        bars = make_bars()
        dates = bars.index.get_level_values("timestamp").unique().sort_values()
        missing_date = dates[300]
        bars.loc[("AAPL", missing_date), "close"] = np.nan

        with self.assertRaisesRegex(ValueError, rf"AAPL.*{missing_date.date()}"):
            monthly_momentum_targets(bars)

    def test_missing_execution_open_fails_with_symbol_and_date(self):
        bars = make_bars()
        dates = bars.index.get_level_values("timestamp").unique().sort_values()
        first_signal_position = next(
            position
            for position in month_end_positions(dates)
            if position >= 252 and position + 1 < len(dates)
        )
        execution_date = dates[first_signal_position + 1]
        bars.loc[("AAPL", execution_date), "open"] = np.nan

        with self.assertRaisesRegex(ValueError, rf"AAPL.*{execution_date.date()}"):
            monthly_momentum_targets(bars)


class PortfolioSimulationTests(unittest.TestCase):
    def test_rebalances_at_execution_opens_and_charges_slippage_per_side(self):
        bars = make_bars(session_count=4)
        dates = bars.index.get_level_values("timestamp").unique().sort_values()
        first_execution, second_execution = dates[1], dates[2]
        bars.loc[(TICKERS[0], first_execution), "close"] = 110.0
        bars.loc[(TICKERS[0], second_execution), ["open", "close"]] = 110.0
        targets = make_targets(
            [first_execution, second_execution], [TICKERS[:10], TICKERS[10:20]]
        )

        equity, summary = simulate_portfolio(bars, targets)

        rate = 5.0 / 10_000
        first_invested_value = 100_000.0 / (1.0 + rate)
        expected_first_close = first_invested_value * 1.01
        self.assertAlmostEqual(equity.loc[first_execution], expected_first_close, places=6)
        self.assertAlmostEqual(summary["rebalance_count"], 2)
        self.assertGreater(summary["gross_turnover"], 2.9)
        self.assertAlmostEqual(
            summary["slippage_dollars"],
            100_000.0 - first_invested_value
            + rate
            * (
                expected_first_close
                + expected_first_close * (1.0 - rate) / (1.0 + rate)
            ),
            places=5,
        )

    def test_equal_weight_benchmark_buys_once_at_same_execution_open(self):
        bars = make_bars(session_count=4)
        dates = bars.index.get_level_values("timestamp").unique().sort_values()
        execution_date = dates[1]
        for date in dates[1:]:
            bars.loc[(TICKERS[0], date), "close"] = 110.0

        equity = simulate_equal_weight_benchmark(
            bars, execution_date, dates[-1]
        )

        self.assertEqual(equity.index[0], execution_date)
        expected_invested_value = 100_000.0 / (1.0 + 5.0 / 10_000)
        expected_first_close = expected_invested_value * (1.0 + 0.10 / len(TICKERS))
        self.assertAlmostEqual(equity.iloc[0], expected_first_close, places=6)
        self.assertAlmostEqual(equity.iloc[-1], expected_first_close, places=6)

    def test_metrics_include_initial_capital_as_starting_peak(self):
        dates = pd.bdate_range("2025-01-01", periods=3)
        equity = pd.Series([110.0, 100.0, 121.0], index=dates)

        metrics = calculate_metrics(equity, initial_capital=100.0)

        returns = np.array([0.10, 100.0 / 110.0 - 1.0, 121.0 / 100.0 - 1.0])
        self.assertAlmostEqual(metrics["total_return_pct"], 21.0)
        expected_annualized = (1.21 ** (252.0 / 3.0) - 1.0) * 100.0
        self.assertAlmostEqual(metrics["annualized_return_pct"], expected_annualized)
        self.assertAlmostEqual(
            metrics["annualized_sharpe"],
            returns.mean() / returns.std(ddof=1) * np.sqrt(252.0),
        )
        self.assertAlmostEqual(metrics["max_drawdown_pct"], (100.0 / 110.0 - 1.0) * 100.0)


class BacktestRunnerTests(unittest.TestCase):
    def test_runner_returns_aligned_equity_and_excess_return_metrics(self):
        bars = make_bars()

        curves, results = run_backtest(bars)

        self.assertEqual(
            list(curves.columns), ["candidate_equity", "benchmark_equity"]
        )
        self.assertTrue(curves.index.equals(curves.dropna().index))
        self.assertEqual(curves["candidate_equity"].first_valid_index(),
                         curves["benchmark_equity"].first_valid_index())
        self.assertEqual(curves["candidate_equity"].last_valid_index(),
                         curves["benchmark_equity"].last_valid_index())
        self.assertAlmostEqual(
            results["excess_cumulative_return_pct"],
            results["candidate"]["total_return_pct"]
            - results["benchmark"]["total_return_pct"],
        )
        self.assertAlmostEqual(
            results["excess_annualized_return_pct"],
            results["candidate"]["annualized_return_pct"]
            - results["benchmark"]["annualized_return_pct"],
        )
        self.assertIn("rebalance_count", results["candidate"])
        self.assertIn("gross_turnover", results["candidate"])
        self.assertIn("slippage_dollars", results["candidate"])
        self.assertAlmostEqual(
            results["benchmark"]["gross_turnover"],
            1.0 / (1.0 + 5.0 / 10_000),
        )


if __name__ == "__main__":
    unittest.main()

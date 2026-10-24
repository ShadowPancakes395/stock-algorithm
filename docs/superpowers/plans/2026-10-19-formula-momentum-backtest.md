# Formula Momentum Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated, reproducible backtest for the approved 12–1 momentum strategy and compare it fairly with equal-weight buy-and-hold.

**Architecture:** Add one standalone module, `momentum_backtest.py`, with pure target generation, portfolio simulation, benchmark simulation, metrics, and a command-line entry point. Add one standard-library test module, `test_momentum_backtest.py`, using synthetic daily bars so core math can be verified without broker access. The command-line path may fetch historical bars with the existing `data.fetch_daily_bars`; it must never import or invoke order-submission code.

**Tech Stack:** Python, pandas, numpy, Alpaca historical-data client already used by `data.py`, Python `unittest`.

## Global Constraints

- Universe: the existing 28 tickers from `config.py`.
- Direction: long-only, no leverage, no short positions.
- Signal date: the final trading session of each calendar month.
- Selection: rank eligible tickers by score, descending; select the top 10.
- Portfolio weights: equal weight across the selected 10, 10% each. The strategy remains fully invested in those holdings.
- Rebalance: monthly. Replace a holding when it leaves the selected top 10 and buy a newly selected name when one enters.
- Execution: use the next trading session's adjusted open after the signal close. Do not execute from the signal-date close, which would use information before an order could have been placed.
- Trading costs: apply 5 basis points of slippage to each side of every rebalance trade, consistent with the project's existing estimate. Apply the same cost convention to initial benchmark purchases.
- Risk controls: do not use RSI, the volatility forecast, volatility-scaled sizing, or stop-loss exits in this first candidate.
- The existing live strategy, `trade.py`, `rules.py`, `check_protection.py`, scheduler configuration, and broker account are out of scope and must remain unchanged.

---

## File map

- Create `momentum_backtest.py`: deterministic monthly momentum targets, strict input validation, next-session-open portfolio accounting, equal-weight buy-and-hold benchmark, metric calculations, and CLI/reporting.
- Create `test_momentum_backtest.py`: synthetic-bar tests for exact score formula, month-end selection, top-10 weights, data validation, execution timing, slippage, benchmark timing, and metrics.
- Read only `config.py`, `data.py`, and the approved design spec. No other application files need modification.

## Task 1: Momentum targets and input validation

**Files:**
- Create: `test_momentum_backtest.py`
- Create: `momentum_backtest.py`

**Interfaces:**
- `monthly_momentum_targets(bars: pd.DataFrame) -> pd.DataFrame`
- Input: daily OHLCV `DataFrame` with `(symbol, timestamp)` MultiIndex and `open`, `close` columns, as returned by `data.fetch_daily_bars`.
- Output: rows indexed by `(execution_date, symbol)` with `signal_date`, `score`, and `target_weight` columns. Each monthly execution date contains exactly 10 rows weighted 0.10 each.
- Constants: `MOMENTUM_LOOKBACK_SESSIONS = 252`, `MOMENTUM_SKIP_SESSIONS = 21`, `TOP_N = 10`.

- [ ] **Step 1: Write failing signal tests**

Create synthetic bars for 28 symbols over at least 300 shared business sessions. Include one test where the top-10 ranking is known from close prices at `t-252` and `t-21`; another verifies signals occur on each month’s last available session and are assigned to the following session; another verifies 10 selected symbols with 10% weights. Add tests that fewer than 253 sessions produce no executable target, a missing required close fails with the symbol/date, and a missing next-session open fails with the symbol/date.

- [ ] **Step 2: Run tests and confirm the new API is absent**

Run: `venv\Scripts\python.exe -m unittest -v test_momentum_backtest`
Expected: FAIL because `momentum_backtest` and `monthly_momentum_targets` do not exist.

- [ ] **Step 3: Implement the minimal target generator**

Build aligned close/open matrices in the fixed configured ticker order. Validate positive finite closes across the full required history and valid positive opens on all monthly execution dates; raise `ValueError` naming the first symbol and date with missing/invalid data. Identify each calendar month’s last available session. For a signal at row position `i`, compute `close[i - 21] / close[i - 252] - 1`. Skip signal dates before the 252-session history exists. Rank descending, breaking exact ties by ticker symbol for deterministic results. Emit top 10 weights for the next shared session only; do not emit an unexecutable target when there is no following session.

- [ ] **Step 4: Run the signal tests**

Run: `venv\Scripts\python.exe -m unittest -v test_momentum_backtest`
Expected: all Task 1 signal and validation tests PASS.

## Task 2: Next-open simulation, benchmark, and metrics

**Files:**
- Modify: `test_momentum_backtest.py`
- Modify: `momentum_backtest.py`

**Interfaces:**
- `simulate_portfolio(bars: pd.DataFrame, targets: pd.DataFrame, initial_capital: float = 100000.0, slippage_bps: float = 5.0) -> tuple[pd.Series, dict]`
- `simulate_equal_weight_benchmark(bars: pd.DataFrame, start_date, end_date, initial_capital: float = 100000.0, slippage_bps: float = 5.0) -> pd.Series`
- `calculate_metrics(equity: pd.Series) -> dict`
- Portfolio equity output: daily close-valued `Series`; simulator summary includes cumulative and annualized returns, rebalance count, gross turnover, and dollar slippage.

- [ ] **Step 1: Write failing portfolio tests**

Use a small synthetic price set and hand-computed open/close prices to prove that (a) a signal dated `d` trades only at the next session’s open, (b) no holding is purchased at the signal-date close, (c) buy and sell fills charge 5 bps per side, (d) equal-weight benchmark buys at the same first execution open and holds, and (e) equity, cumulative return, annualized return, Sharpe, and maximum drawdown match hand-calculated values. Verify gross turnover is traded notional divided by pre-trade equity and reported slippage equals traded notional times 0.0005.

- [ ] **Step 2: Run tests and confirm simulator functions are absent**

Run: `venv\Scripts\python.exe -m unittest -v test_momentum_backtest`
Expected: FAIL for the new simulator and metric APIs.

- [ ] **Step 3: Implement deterministic accounting**

Initialize the candidate at its first target execution date with $100,000 cash. At each execution open, mark existing holdings to that open, calculate pre-trade equity, trade to the new target weights, apply adverse 5 bps fill adjustment to every buy and sell, and then mark all positions at each session close. Keep cash nonnegative within floating-point tolerance; fail if the target set or execution prices make the portfolio infeasible. Start the benchmark with an equal-weight purchase across all 28 tickers at the same first execution open, apply the same buy-side slippage, and hold without rebalancing. Use identical valuation dates. Calculate cumulative return, annualized return using 252 sessions per year, annualized Sharpe from daily close-to-close equity returns, and maximum drawdown from the close-valued equity curve. Count each executed monthly date as a rebalance and aggregate traded notional and slippage dollars.

- [ ] **Step 4: Run portfolio and metric tests**

Run: `venv\Scripts\python.exe -m unittest -v test_momentum_backtest`
Expected: all Task 1 and Task 2 tests PASS.

## Task 3: CLI report and equity-curve output

**Files:**
- Modify: `test_momentum_backtest.py`
- Modify: `momentum_backtest.py`

**Interfaces:**
- `run_backtest(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]`
- CLI: `python momentum_backtest.py`
- Output file: `momentum_backtest_equity_curve.csv`, containing aligned date, candidate equity, and benchmark equity columns.

- [ ] **Step 1: Add a runner test**

Patch `data.fetch_daily_bars` in a test to return deterministic synthetic bars; call `run_backtest(bars)` and assert the result includes candidate and benchmark equity on identical dates, all requested metrics, and a percentage-point excess-return field calculated as candidate annualized return minus benchmark annualized return. Keep the test independent of the Alpaca API and do not write the production output CSV from tests.

- [ ] **Step 2: Implement the isolated runner and CLI**

`run_backtest` must call the target generator, candidate simulator, and benchmark simulator, align their equity curves, and return a combined frame plus metrics for both portfolios and their differences. Report cumulative and annualized percentage-point excess returns. The CLI fetches daily bars once, prints cumulative/annualized/excess returns, max drawdown, Sharpe, rebalance count, turnover, and slippage, then writes the combined equity curve CSV. It must not import `trade.py`, instantiate `TradingClient`, or submit orders. Print the fixed lookback, skip, top-N, and slippage parameters with the report. Include the survivorship/universe-selection caveat in the CLI report.

- [ ] **Step 3: Run unit tests**

Run: `venv\Scripts\python.exe -m unittest -v test_momentum_backtest`
Expected: all tests PASS without network access or changes to live-trading files.

- [ ] **Step 4: Run the historical backtest**

Run: `venv\Scripts\python.exe momentum_backtest.py`
Expected: one report for candidate and benchmark over identical dates, with metrics and a generated `momentum_backtest_equity_curve.csv`. If the historical-data request cannot complete, preserve all source files and report the API/network failure separately from code/test results.

## Self-review against the spec

- Fixed formula, top 10, monthly schedule, next-session execution, slippage, risk-control exclusions, comparison benchmark, metrics, output curve, and survivorship caveat are assigned to Tasks 1–3.
- Data errors fail with symbol/date instead of silently filling prices or changing the universe.
- No implementation task touches the existing live strategy or scheduler.
- All test commands use the repo virtual environment and `unittest`; no test dependency is added.

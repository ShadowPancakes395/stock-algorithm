# Formula Momentum Backtest Design

## Status

Design approved by the user on 2026-10-19; awaiting review of this written specification before implementation.

## Objective

Evaluate whether a fixed, formula-based momentum strategy can outperform an equal-weight buy-and-hold portfolio on the project's existing 28-stock universe. The user's stated priority is higher returns versus the benchmark, with willingness to accept a moderately higher drawdown.

This is a research backtest only. The existing live paper-trading strategy, its configuration, and its scheduled tasks remain untouched.

## Candidate rule

- Universe: the existing 28 tickers from `config.py`.
- Direction: long-only, no leverage, no short positions.
- Signal date: the final trading session of each calendar month.
- Momentum score for each ticker on signal date `t`: adjusted close at `t - 21 trading sessions` divided by adjusted close at `t - 252 trading sessions`, minus 1. This measures roughly 12-month return while skipping the latest month.
- Selection: rank eligible tickers by score, descending; select the top 10.
- Portfolio weights: equal weight across the selected 10, 10% each. The strategy remains fully invested in those holdings.
- Rebalance: monthly. Replace a holding when it leaves the selected top 10 and buy a newly selected name when one enters.
- Execution: use the next trading session's adjusted open after the signal close. Do not execute from the signal-date close, which would use information before an order could have been placed.
- Trading costs: apply 5 basis points of slippage to each side of every rebalance trade, consistent with the project's existing estimate. Apply the same cost convention to initial benchmark purchases.
- Risk controls: do not use RSI, the volatility forecast, volatility-scaled sizing, or stop-loss exits in this first candidate. Omitting them isolates the effect of momentum selection. The possible increase in drawdown is a measured outcome, not an instruction to increase risk in live trading.

## Comparison and evaluation

The benchmark is equal-weight buy-and-hold across the same 28 tickers. Both portfolios start on the same first eligible execution date, use the same adjusted price series and valuation dates, and account for the same initial slippage convention. The benchmark is not rebalanced after its initial equal-weight purchase.

Use the available five-year daily history. Exclude the initial momentum lookback period. After that, require valid data for every ticker and rebalance execution; stop with a clear data error if any required observation is missing. Keep the formula and top-10 selection fixed; do not tune parameters against the test-period results.

Report:

- Cumulative and annualized return, plus percentage-point excess return versus the benchmark.
- Maximum drawdown and annualized Sharpe ratio.
- Number of rebalance dates, turnover, and estimated total slippage.
- The equity curves for the candidate and benchmark on identical dates.

The primary success measure is positive annualized excess return versus equal-weight buy-and-hold over the full evaluated period. Maximum drawdown is a key trade-off measure because the user accepts moderate, but not unlimited, increases in drawdown. A single historical result does not establish that the strategy will continue to outperform.

## Method and scope

Implement the candidate as a separate backtest path. Reuse the project's historical daily bars and benchmark universe, but do not route candidate signals into `trade.py` or the paper account. Use fixed parameters and chronological data only; the momentum rule has no model training step.

The existing backtest uses same-day closing prices for signals and simulated trades. That convention is not suitable for this candidate. The candidate and benchmark must be evaluated using the next-session execution basis described above. The existing strategy's historical result should not be presented as a directly comparable result unless its execution timing is separately corrected and both strategies are rerun on matching dates and costs.

## Data checks and limitations

- Require valid adjusted prices for all 28 symbols over the full signal lookback and valid adjusted opens on each next-session rebalance date. If data is missing, stop with the affected symbol and date; do not fill prices, substitute a same-day close, or silently change the universe.
- The historical 28-stock basket reflects the current configured universe; it may not represent the constituents that would have been selected at each historical date. Report this survivorship/universe-selection limitation.
- The 5 bps slippage estimate is a simplifying assumption, not a fill guarantee. The backtest does not prove live or paper execution quality.
- Do not change any live strategy file, scheduler configuration, broker account, or order during this work.

## Acceptance criteria

1. Candidate logic is separate from the current live rules and does not submit orders.
2. Momentum uses only prices available before the next-session execution.
3. The top 10 are selected monthly by the fixed 12-month-excluding-latest-month formula and are equally weighted.
4. Candidate and benchmark use the same dates, valuation basis, and specified slippage treatment.
5. Results include all requested return, risk, turnover, and cost measures, plus the stated data limitations.
6. No strategy parameter is changed after inspecting the test results.

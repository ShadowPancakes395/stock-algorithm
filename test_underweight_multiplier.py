"""TEST ONLY -- does not modify rules.py or the live system.

Cleanly isolates UNDERWEIGHT_MULTIPLIER as the single variable, unlike the
earlier 9/18 test where it was bundled with a slippage-modeling change in
the same run (never actually isolated -- see chat history). Both legs
here share one data pull, so the benchmark is identical across both,
avoiding the apples-to-oranges problem from comparing separate runs on
different days.

Motivated by live evidence: UNDERWEIGHT_MULTIPLIER=0.3 appears to have
cost real relative performance during the market rally following the
9/18 stop-loss floor fix (strategy captured ~1/3 of the benchmark's gain
over that period) -- consistent with the mechanism (trimming exposure to
overbought/momentum names loses more in a rally than it saves).
"""
import pandas as pd

import rules
from data import fetch_daily_bars
from features import add_features
from model import FEATURE_COLS, build_dataset, train_model
from backtest import walk_forward_folds, simulate_portfolio, compute_metrics, benchmark_equal_weight

CANDIDATES = [0.3, 0.5, 0.6]  # 0.3 is current live value, the others are candidates


def run_backtest_with_multiplier(dataset: pd.DataFrame, multiplier: float):
    rules.UNDERWEIGHT_MULTIPLIER = multiplier  # monkeypatch the module constant

    folds = walk_forward_folds(dataset)
    all_signals = []
    for train, test in folds:
        model = train_model(train)
        forecast = pd.Series(
            model.predict(test[FEATURE_COLS]), index=test.index, name="forecast_volatility"
        )
        signals = rules.generate_signals(test, forecast)
        all_signals.append(signals)
    return pd.concat(all_signals).sort_index(level="timestamp")


if __name__ == "__main__":
    print("Fetching data and building features (once, shared across all candidates)...")
    bars = fetch_daily_bars()
    featured = add_features(bars)
    dataset = build_dataset(featured)

    results = {}
    for mult in CANDIDATES:
        print(f"\n=== UNDERWEIGHT_MULTIPLIER = {mult} ===")
        signals = run_backtest_with_multiplier(dataset, mult)
        equity, trades = simulate_portfolio(signals, bars["close"])
        metrics = compute_metrics(equity, trades)
        results[mult] = metrics
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== BENCHMARK (same for all candidates, same data pull) ===")
    start_date = dataset.index.get_level_values("timestamp").min()
    end_date = dataset.index.get_level_values("timestamp").max()
    bench = benchmark_equal_weight(bars, start_date, end_date)
    for k, v in bench.items():
        print(f"  {k}: {v:.3f}")

    print("\n=== SUMMARY ===")
    for mult, m in results.items():
        print(f"  mult={mult}: return={m['total_return_pct']:.2f}%  "
              f"sharpe={m['annualized_sharpe']:.3f}  "
              f"max_dd={m['max_drawdown_pct']:.2f}%  "
              f"trades={m['num_trades']}")

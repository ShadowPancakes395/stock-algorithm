"""Daily trading script -- generates today's target portfolio weights and
submits orders to Alpaca (paper trading) to move current positions toward
them. Designed to be run once per trading day via Windows Task Scheduler.

SAFETY: refuses to run unless ALPACA_PAPER=true in .env. This is a hard
stop, not a warning -- accidentally running this against a live account
is exactly the kind of mistake this check exists to prevent.

Mechanics:
  - Retrains the volatility model on all available history each run
    (cheap at this data size; avoids a silently stale saved model).
  - Computes today's target weight per ticker via rules.py, using only
    data available as of today (no lookahead -- today's close is the most
    recent bar available intraday/after-hours, matching how the backtest
    used same-day close to size same-day positions).
  - Diffs target weights against CURRENT Alpaca positions and submits
    orders only for the difference, not a full daily liquidate/rebuild.
  - Stop-losses are placed as real Alpaca stop orders at entry time
    (continuously enforced by Alpaca, not just checked once/day like the
    backtest's simplified simulation -- strictly better, not a shortcut).
"""
import sys
import time
import argparse
from datetime import datetime

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.enums import DataFeed

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER, TICKERS
from data import fetch_daily_bars
from features import add_features
from model import FEATURE_COLS, build_dataset, train_model
from rules import generate_signals

LOG_PATH = "trade_log.txt"


def log(msg: str):
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def get_todays_targets() -> pd.DataFrame:
    """Train on all history with a valid target, generate today's target
    weights using the LATEST available features.

    BUG FIX (caught via --dry-run testing): the original version called
    build_dataset() and predicted on its own output. build_dataset()
    drops any row missing forward_volatility_5d, which is undefined for
    the most recent ~5 trading days (no future returns exist yet to
    compute it from) -- correct for TRAINING data, but it meant live
    predictions were silently generated from ~1 week stale data every
    run, including yesterday's real trades. Fix: train on build_dataset()
    (needs the target), but predict on add_features() output directly
    (only needs FEATURE_COLS, not a target), so today's row is available.
    """
    bars = fetch_daily_bars()
    featured = add_features(bars)

    training_data = build_dataset(featured)  # target-dropna'd, for training only
    model = train_model(training_data)

    # Predict on the full featured set (only needs FEATURE_COLS to be
    # non-null), so today's row -- which has no target yet -- is included.
    predictable = featured.dropna(subset=FEATURE_COLS)
    forecast = pd.Series(
        model.predict(predictable[FEATURE_COLS]), index=predictable.index, name="forecast_volatility"
    )
    signals = generate_signals(predictable, forecast)

    latest_date = signals.index.get_level_values("timestamp").max()
    today_signals = signals.xs(latest_date, level="timestamp")
    log(f"Generated targets using data through {latest_date.date()} "
        f"({len(today_signals)} tickers)")
    return today_signals


def get_current_positions(client: TradingClient) -> dict:
    """Returns {symbol: market_value} for current Alpaca positions."""
    positions = client.get_all_positions()
    return {p.symbol: float(p.market_value) for p in positions}


def rebalance(client: TradingClient, data_client: StockHistoricalDataClient, targets: pd.DataFrame,
              dry_run: bool = False):
    """Diff target weights against current positions and submit orders
    for the difference. Signal SELL (underweight band) with no current
    position is skipped -- never enter fresh into an overbought name.

    BUY orders use whole-share quantities (not notional/fractional):
    fractional-share orders on Alpaca can only be DAY orders, which would
    make any stop-loss on them expire at market close -- no protection
    overnight. Whole shares allow GTC stop-losses, which is required given
    this system holds positions across multiple days, not intraday.

    SAFETY: tracks a running total of committed cash across this run and
    refuses to submit any BUY that would push total exposure past actual
    account equity. Paper accounts default to 4x margin (buying_power far
    exceeds portfolio_value); this is a hard backstop against unintended
    leverage, independent of whether the sizing math elsewhere is correct.

    dry_run: logs intended orders and quantities but never calls
    submit_order -- lets the full pipeline (data, features, model, rules,
    diffing, sizing, margin guard) be exercised outside market hours
    without touching the account.
    """
    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    current = get_current_positions(client)
    committed = sum(current.values())  # running total, starts at current holdings

    for symbol, row in targets.iterrows():
        target_value = portfolio_value * row["position_size_pct"]
        if row["signal"] == "SELL" and symbol not in current:
            continue  # don't open fresh positions in overbought names

        current_value = current.get(symbol, 0.0)
        diff_value = target_value - current_value

        # Skip tiny rebalances -- not worth the trade cost for <1% of
        # portfolio value in drift, avoids churn from noise-level changes.
        if abs(diff_value) < portfolio_value * 0.01:
            continue

        side = OrderSide.BUY if diff_value > 0 else OrderSide.SELL

        if side == OrderSide.BUY:
            quote = data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
            )[symbol]
            price = float(quote.ask_price) if quote.ask_price else float(quote.bid_price)
            qty = int(abs(diff_value) // price)  # whole shares only, round down
            if qty < 1:
                log(f"  SKIPPED {symbol}: target ${abs(diff_value):.2f} buys <1 share at ${price:.2f}")
                continue
            order_value = qty * price
            stop_price = round(price * (1 - row["stop_loss_pct"]), 2)

            projected_committed = committed + order_value
            if projected_committed > portfolio_value:
                log(f"  BLOCKED BUY {symbol}: {qty} shares (~${order_value:.2f}) would push total "
                    f"committed to ${projected_committed:.2f}, exceeding portfolio value "
                    f"${portfolio_value:.2f} (no margin allowed). Skipping.")
                continue
            committed = projected_committed

            if dry_run:
                log(f"  [DRY RUN] would BUY {symbol}: {qty} shares @ ~${price:.2f} "
                    f"(~${order_value:.2f}), stop-loss target ${stop_price:.2f}")
                continue

            try:
                order = client.submit_order(MarketOrderRequest(
                    symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
                ))
                log(f"  BUY {symbol}: {qty} shares (~${order_value:.2f}, order {order.id})")
            except Exception as e:
                log(f"  FAILED BUY {symbol}: {e}")
                continue
            _place_stop_loss(client, order.id, symbol, row["stop_loss_pct"])

        else:
            notional = abs(diff_value)
            committed -= min(notional, current_value)

            if dry_run:
                log(f"  [DRY RUN] would SELL {symbol}: ~${notional:.2f}")
                continue

            try:
                order = client.submit_order(MarketOrderRequest(
                    symbol=symbol, notional=round(notional, 2),
                    side=side, time_in_force=TimeInForce.DAY,
                ))
                log(f"  SELL {symbol}: ${notional:.2f} (order {order.id})")
            except Exception as e:
                log(f"  FAILED SELL {symbol}: {e}")


def _place_stop_loss(client: TradingClient, order_id: str, symbol: str, stop_loss_pct: float,
                      poll_interval_s: float = 2.0, max_wait_s: float = 30.0):
    """Poll the just-submitted BUY order until it fills, then place a stop
    order at (fill_price * (1 - stop_loss_pct)) for the filled quantity.

    Market orders fill almost immediately during market hours, so this
    poll is normally 1-3 iterations. If it doesn't fill within max_wait_s,
    logs a clear warning rather than silently skipping the stop-loss --
    an unprotected position should be loud, not silent.
    """
    elapsed = 0.0
    while elapsed < max_wait_s:
        order = client.get_order_by_id(order_id)
        if order.status == OrderStatus.FILLED:
            fill_price = float(order.filled_avg_price)
            qty = float(order.filled_qty)
            stop_price = round(fill_price * (1 - stop_loss_pct), 2)
            try:
                client.submit_order(StopOrderRequest(
                    symbol=symbol, qty=qty, side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC, stop_price=stop_price,
                ))
                log(f"    stop-loss placed for {symbol}: {qty} shares @ ${stop_price:.2f}")
            except Exception as e:
                log(f"    STOP-LOSS FAILED for {symbol} -- position is UNPROTECTED: {e}")
            return
        time.sleep(poll_interval_s)
        elapsed += poll_interval_s

    log(f"    WARNING: {symbol} BUY order {order_id} did not fill within "
        f"{max_wait_s}s -- no stop-loss placed, check manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Run the full pipeline and log intended orders, but never submit "
                              "real orders. Also bypasses the market-hours check, so the pipeline "
                              "can be exercised for testing even when markets are closed.")
    args = parser.parse_args()

    if not ALPACA_PAPER:
        log("REFUSING TO RUN: ALPACA_PAPER is not True. This script only runs against paper accounts.")
        sys.exit(1)

    log(f"=== Starting daily trading run{' [DRY RUN]' if args.dry_run else ''} ===")
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    if not args.dry_run:
        clock = client.get_clock()
        if not clock.is_open:
            log(f"Market is closed (next open: {clock.next_open}). Exiting without trading.")
            sys.exit(0)

    targets = get_todays_targets()
    rebalance(client, data_client, targets, dry_run=args.dry_run)
    log("=== Run complete ===")

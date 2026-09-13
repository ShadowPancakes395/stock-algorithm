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
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
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
    """Returns {symbol: {"market_value": ..., "available_qty": ...}} for
    current Alpaca positions. available_qty excludes shares already held
    for pending orders (e.g. a GTC stop-loss) -- needed because SELLing
    more than the unheld quantity gets rejected outright.
    """
    positions = client.get_all_positions()
    return {
        p.symbol: {
            "market_value": float(p.market_value),
            "available_qty": float(p.qty_available),
        }
        for p in positions
    }


def get_open_orders_by_symbol(client: TradingClient) -> dict:
    """{symbol: order} for the current open (unfilled) order on each
    symbol -- lets rebalance() cancel a symbol's stale stop-loss before
    submitting a new order for that symbol, rather than skip the trade
    entirely (see trade.py module docstring: skipping left the whole
    portfolio "frozen" from further rebalancing within days of the first
    real run, since nearly every symbol accumulates a standing GTC stop).
    Assumes at most one open order per symbol, true by construction here
    (rebalance() only ever creates one stop per position).
    """
    open_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
    return {o.symbol: o for o in open_orders}


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

    Symbols with an existing open order (a standing GTC stop-loss from a
    prior BUY) have that order CANCELED before a new BUY/SELL is
    submitted, then a fresh stop-loss is placed after the new order fills.
    Earlier versions skipped these entirely, which -- given nearly every
    symbol accumulates a stop within days of the first real run -- ended
    up freezing the whole portfolio from further rebalancing. Canceling
    and replacing keeps the position genuinely actively managed, matching
    what was actually backtested.
    """
    account = client.get_account()
    portfolio_value = float(account.portfolio_value)
    current = get_current_positions(client)
    open_orders_by_symbol = get_open_orders_by_symbol(client)
    committed = sum(p["market_value"] for p in current.values())  # running total, starts at current holdings

    for symbol, row in targets.iterrows():
        target_value = portfolio_value * row["position_size_pct"]
        if row["signal"] == "SELL" and symbol not in current:
            continue  # don't open fresh positions in overbought names

        current_value = current.get(symbol, {}).get("market_value", 0.0)
        diff_value = target_value - current_value

        # Skip tiny rebalances -- not worth the trade cost for <1% of
        # portfolio value in drift, avoids churn from noise-level changes.
        if abs(diff_value) < portfolio_value * 0.01:
            continue

        side = OrderSide.BUY if diff_value > 0 else OrderSide.SELL

        if symbol in open_orders_by_symbol:
            if dry_run:
                log(f"  [DRY RUN] would CANCEL existing stop-loss order on {symbol} before rebalancing")
            else:
                try:
                    client.cancel_order_by_id(open_orders_by_symbol[symbol].id)
                    log(f"  Canceled existing stop-loss on {symbol} to allow rebalance")
                except Exception as e:
                    log(f"  FAILED to cancel existing order on {symbol}, skipping rebalance for it: {e}")
                    continue

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
            # available_qty may be stale immediately after a cancel (the
            # freed shares can take a moment to reflect) -- re-fetch this
            # symbol's position fresh rather than trust the pre-cancel snapshot.
            fresh = client.get_all_positions()
            fresh_available = next((float(p.qty_available) for p in fresh if p.symbol == symbol), 0.0)

            if fresh_available <= 0:
                log(f"  SKIPPED SELL {symbol}: no available (unheld) shares to sell even after cancel.")
                continue

            quote = data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
            )[symbol]
            price = float(quote.bid_price) if quote.bid_price else float(quote.ask_price)
            desired_qty = abs(diff_value) / price
            sell_qty = min(desired_qty, fresh_available)
            notional_value = sell_qty * price
            committed -= min(notional_value, current_value)

            if dry_run:
                log(f"  [DRY RUN] would SELL {symbol}: {sell_qty:.4f} shares (~${notional_value:.2f})")
                continue

            try:
                order = client.submit_order(MarketOrderRequest(
                    symbol=symbol, qty=round(sell_qty, 4),
                    side=side, time_in_force=TimeInForce.DAY,
                ))
                log(f"  SELL {symbol}: {sell_qty:.4f} shares (~${notional_value:.2f}, order {order.id})")
            except Exception as e:
                log(f"  FAILED SELL {symbol}: {e}")
                continue

            # Re-protect whatever's left of the position with a fresh stop.
            # BUG FIX #1: previously computed remaining_qty as
            # (fresh_available - sell_qty), i.e. remaining AVAILABLE qty --
            # but fresh_available was already just the unheld sliver before
            # this sell (the bulk of the position was locked under the
            # just-canceled stop). That silently skipped re-protecting the
            # bulk of the position whenever a sell only touched a small
            # unheld remainder. Caught live: JPM held ~17 shares, a sell of
            # 0.87 unheld shares left "remaining_qty=0" by the old logic,
            # so no stop was re-placed and all 17 shares sat unprotected.
            # BUG FIX #2: the first fix used a flat time.sleep(2) before
            # reading position qty, assuming the sell would have settled by
            # then. Caught live again the next day: the sleep wasn't long
            # enough, the stop submission read a qty that still included
            # shares held_for_orders from the just-filled sell, and Alpaca
            # rejected the stop for requesting more than was available --
            # leaving the position unprotected a second time via a
            # different path than bug #1. Fix: poll get_order_by_id for
            # this order to reach FILLED, same pattern _place_stop_loss
            # already uses for the BUY side, instead of guessing a sleep
            # duration.
            filled = _wait_for_fill(client, order.id)
            if not filled:
                log(f"    WARNING: {symbol} SELL order {order.id} did not confirm filled -- "
                    f"skipping stop re-placement, check manually.")
                continue

            post_sell = client.get_all_positions()
            remaining_qty = next((float(p.qty) for p in post_sell if p.symbol == symbol), 0.0)

            if remaining_qty > 0:
                whole_remaining = int(remaining_qty)
                if whole_remaining >= 1:
                    stop_price = round(price * (1 - row["stop_loss_pct"]), 2)
                    try:
                        client.submit_order(StopOrderRequest(
                            symbol=symbol, qty=whole_remaining, side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC, stop_price=stop_price,
                        ))
                        log(f"    re-placed stop-loss for {symbol}: {whole_remaining} shares "
                            f"(full remaining position) @ ${stop_price:.2f}")
                    except Exception as e:
                        log(f"    FAILED to re-place stop-loss for {symbol} -- remaining position "
                            f"is UNPROTECTED: {e}")


def _wait_for_fill(client: TradingClient, order_id: str,
                    poll_interval_s: float = 2.0, max_wait_s: float = 30.0) -> bool:
    """Poll an order until it reaches FILLED status, or return False after
    max_wait_s. Shared by _place_stop_loss (BUY side) and rebalance's SELL
    branch -- both need to know an order has genuinely settled before
    trusting a subsequent position-quantity read, not just guess a sleep
    duration (see rebalance() SELL branch docstring for why a fixed sleep
    wasn't reliable).
    """
    elapsed = 0.0
    while elapsed < max_wait_s:
        order = client.get_order_by_id(order_id)
        if order.status == OrderStatus.FILLED:
            return True
        time.sleep(poll_interval_s)
        elapsed += poll_interval_s
    return False


def _place_stop_loss(client: TradingClient, order_id: str, symbol: str, stop_loss_pct: float):
    """Wait for the just-submitted BUY order to fill, then place a stop
    order sized to the symbol's TOTAL current position, not just this
    order's filled_qty.

    BUG FIX: when a symbol already held shares (e.g. its prior stop-loss
    was just canceled to allow this BUY -- see rebalance()), sizing the
    new stop to only the newly-filled quantity left the pre-existing
    shares completely unprotected. Caught live: LIN held 9.28 shares but
    the stop only covered the 2 just bought, after a cancel-and-rebuy.
    Fix: query the account's actual current qty for this symbol after
    the fill, and protect that whole amount.

    Market orders fill almost immediately during market hours, so this
    poll is normally 1-3 iterations. If it doesn't fill within max_wait_s,
    logs a clear warning rather than silently skipping the stop-loss --
    an unprotected position should be loud, not silent.

    qty is floored to a whole share: Alpaca's paper engine can report a
    fractional total qty (this is exactly how the original notional-order
    bug produced fractional positions, and legacy fractional remainders
    can persist). A fractional qty would hit the same "fractional orders
    must be DAY orders" rejection GTC stops already failed on once --
    flooring keeps the GTC stop valid; any fractional remainder is
    logged, not silently dropped.
    """
    if not _wait_for_fill(client, order_id):
        log(f"    WARNING: {symbol} BUY order {order_id} did not fill in time -- "
            f"no stop-loss placed, check manually.")
        return

    order = client.get_order_by_id(order_id)
    fill_price = float(order.filled_avg_price)

    # Total current position, not just this order's filled_qty --
    # covers the case where the symbol already held shares.
    positions = client.get_all_positions()
    total_qty = next((float(p.qty) for p in positions if p.symbol == symbol), float(order.filled_qty))

    qty = int(total_qty)  # floor to whole shares -- GTC requirement
    remainder = total_qty - qty
    if remainder > 0:
        log(f"    NOTE: {symbol} total position {total_qty} shares (fractional) -- "
            f"stop-loss covers {qty} whole shares, {remainder:.6f} remainder is UNPROTECTED.")
    if qty < 1:
        log(f"    STOP-LOSS SKIPPED for {symbol}: total qty {total_qty} is entirely "
            f"fractional -- position is UNPROTECTED.")
        return
    stop_price = round(fill_price * (1 - stop_loss_pct), 2)
    try:
        client.submit_order(StopOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC, stop_price=stop_price,
        ))
        log(f"    stop-loss placed for {symbol}: {qty} shares (full position) @ ${stop_price:.2f}")
    except Exception as e:
        log(f"    STOP-LOSS FAILED for {symbol} -- position is UNPROTECTED: {e}")


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

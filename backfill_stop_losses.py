"""ONE-OFF: backfill GTC stop-losses on positions opened by the buggy
2026-08-26 run (fractional-share orders that couldn't carry GTC stops).
Not part of the regular pipeline -- run once, then this script is done.

Pulls each position's actual fill price from Alpaca's own order/position
records (not re-derived), computes the same stop_loss_pct the model
assigned that day via rules.py, and places a GTC stop order for the
position's current (whole-number, if fractional then rounds down) share
count -- since fractional shares still can't carry a GTC stop, any
fractional remainder is left unprotected and reported explicitly rather
than silently ignored.
"""
import math

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
from data import fetch_daily_bars
from features import add_features
from model import FEATURE_COLS, build_dataset, train_model
from rules import generate_signals

assert ALPACA_PAPER, "Refusing to run against a non-paper account."

client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# Recompute today's stop_loss_pct per symbol using the same logic trade.py
# used at 3pm -- deterministic given the same input data, so this
# reproduces the intended stop distance rather than guessing a new one.
bars = fetch_daily_bars()
featured = add_features(bars)
dataset = build_dataset(featured)
model = train_model(dataset)
import pandas as pd
forecast = pd.Series(model.predict(dataset[FEATURE_COLS]), index=dataset.index)
signals = generate_signals(dataset, forecast)
latest_date = signals.index.get_level_values("timestamp").max()
today_signals = signals.xs(latest_date, level="timestamp")

positions = client.get_all_positions()
print(f"Found {len(positions)} open positions. Checking for existing stop orders...")

existing_stops = {
    o.symbol for o in client.get_orders()
    if o.order_type == "stop" and o.status.value in ("new", "accepted", "held")
}

placed, skipped_fractional, skipped_already_protected, skipped_no_signal = [], [], [], []

for pos in positions:
    symbol = pos.symbol
    if symbol in existing_stops:
        skipped_already_protected.append(symbol)
        continue
    if symbol not in today_signals.index:
        skipped_no_signal.append(symbol)
        continue

    qty = math.floor(float(pos.qty))  # whole shares only -- GTC requirement
    if qty < 1:
        skipped_fractional.append(symbol)
        continue

    fill_price = float(pos.avg_entry_price)
    stop_loss_pct = float(today_signals.loc[symbol, "stop_loss_pct"])
    stop_price = round(fill_price * (1 - stop_loss_pct), 2)

    try:
        client.submit_order(StopOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC, stop_price=stop_price,
        ))
        print(f"  {symbol}: stop placed, {qty} shares @ ${stop_price:.2f} "
              f"(fill was ${fill_price:.2f})")
        placed.append(symbol)
    except Exception as e:
        print(f"  {symbol}: FAILED -- {e}")

print()
print(f"Placed: {len(placed)}")
print(f"Already protected (skipped): {skipped_already_protected}")
print(f"No signal data found (skipped): {skipped_no_signal}")
print(f"Fractional-only position, <1 whole share (still unprotected): {skipped_fractional}")

"""ONE-OFF: widen existing stop-losses to the new 8% floor, but only for
positions whose CURRENT stop is narrower than 8% -- symbols where the
volatility-scaled distance was already wider (JPM, CAT, NVDA observed)
are left untouched, since the code change was a floor (a minimum), not a
cap, and tightening those would move in the wrong direction from what
the backtest just validated.

Uses the same poll-until-confirmed cancel pattern trade.py's
_wait_for_cancel established -- a successful cancel_order_by_id call
does not mean Alpaca's backend has finished processing it yet.
"""
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, StopOrderRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce, OrderStatus

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

NEW_FLOOR_PCT = 0.08

assert ALPACA_PAPER, "Refusing to run against a non-paper account."

client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


def wait_for_cancel(order_id: str, poll_s: float = 1.0, max_wait_s: float = 15.0) -> bool:
    terminal = {OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED}
    elapsed = 0.0
    while elapsed < max_wait_s:
        order = client.get_order_by_id(order_id)
        if order.status in terminal:
            return True
        time.sleep(poll_s)
        elapsed += poll_s
    return False


positions = {p.symbol: p for p in client.get_all_positions()}
stops = {o.symbol: o for o in client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))}

updated, left_alone, skipped_no_stop = [], [], []

for symbol, pos in positions.items():
    stop = stops.get(symbol)
    if stop is None:
        skipped_no_stop.append(symbol)
        continue

    entry_price = float(pos.avg_entry_price)
    old_stop_price = float(stop.stop_price)
    old_pct = 1 - old_stop_price / entry_price

    if old_pct >= NEW_FLOOR_PCT:
        left_alone.append((symbol, round(old_pct * 100, 2)))
        continue

    qty = int(float(pos.qty))
    if qty < 1:
        skipped_no_stop.append(symbol)
        continue

    client.cancel_order_by_id(stop.id)
    if not wait_for_cancel(stop.id):
        print(f"  {symbol}: cancel did not confirm in time, skipping to avoid a collision")
        continue

    new_stop_price = round(entry_price * (1 - NEW_FLOOR_PCT), 2)
    client.submit_order(StopOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC, stop_price=new_stop_price,
    ))
    print(f"  {symbol}: widened {round(old_pct*100,2)}% -> 8% "
          f"(${old_stop_price:.2f} -> ${new_stop_price:.2f}), qty={qty}")
    updated.append(symbol)

print()
print(f"Updated: {len(updated)} -> {updated}")
print(f"Left alone (already >= 8%): {left_alone}")
print(f"Skipped (no stop / no whole shares): {skipped_no_stop}")

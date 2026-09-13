"""Standalone daily safety net: verifies every open position has an
adequate stop-loss order protecting it. Deliberately independent of
trade.py's internal position/order tracking logic -- if trade.py has a
bug in how it reasons about positions (as it has, twice: LIN/META/MSFT
after a BUY-side cancel-rebuy, then JPM after a SELL-side cancel-rebuy),
a safety net built from the same assumptions could share the same blind
spot. This queries Alpaca directly and compares total position qty
against total protected qty, nothing else.

Run this daily, right after trade.py, via the same Task Scheduler job.
Exits with a non-zero status code on any gap, so Task Scheduler can be
configured to flag/alert on failure -- not just a log line to notice
later.
"""
import sys
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER

LOG_PATH = "protection_check_log.txt"
GAP_THRESHOLD_SHARES = 1.0  # sub-1-share gaps are expected fractional remainders (see trade.py)


def log(msg: str):
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def check_all_positions_protected(client: TradingClient) -> list:
    """Returns a list of (symbol, total_qty, protected_qty, gap) tuples
    for any position with a gap exceeding GAP_THRESHOLD_SHARES.
    """
    positions = {p.symbol: float(p.qty) for p in client.get_all_positions()}
    open_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
    stop_orders = {o.symbol: float(o.qty) for o in open_orders if o.order_type.value == "stop"}

    gaps = []
    for symbol, total_qty in positions.items():
        protected_qty = stop_orders.get(symbol, 0.0)
        gap = total_qty - protected_qty
        if gap > GAP_THRESHOLD_SHARES:
            gaps.append((symbol, total_qty, protected_qty, gap))
    return gaps


if __name__ == "__main__":
    assert ALPACA_PAPER, "Refusing to run against a non-paper account."

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    gaps = check_all_positions_protected(client)

    if not gaps:
        log("OK: all positions adequately protected.")
        sys.exit(0)

    log(f"!!! PROTECTION GAP DETECTED: {len(gaps)} position(s) with >1 share unprotected !!!")
    for symbol, total, protected, gap in gaps:
        log(f"  {symbol}: total={total:.3f} protected={protected:.3f} gap={gap:.3f}")
    log("Manual intervention needed -- see backfill_stop_losses.py pattern in chat history "
        "for how to fix a specific symbol's gap.")
    sys.exit(1)

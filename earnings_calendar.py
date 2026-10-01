"""Fetch known, dated event risk (earnings releases, pending mergers) for
the ticker basket. Used by rules.py to suppress NEW overweight/buy signals
on a stock right before a known volatility catalyst -- not to force exits
from existing positions, since the goal is avoiding fresh exposure into a
known event, not reacting emotionally to it.

Two sources, since neither alone covers both event types:
  - Earnings dates: Finnhub's free /calendar/earnings endpoint (Alpaca's
    corporate-actions API does NOT include earnings -- confirmed via docs,
    it only covers dividend/merger/spinoff/split).
  - Mergers: Alpaca's own corporate-actions endpoint (already have
    credentials, no new integration needed).
"""
from datetime import date, timedelta

import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCorporateAnnouncementsRequest
from alpaca.trading.enums import CorporateActionType, CorporateActionSubType

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, FINNHUB_API_KEY, TICKERS

FINNHUB_EARNINGS_URL = "https://finnhub.io/api/v1/calendar/earnings"
LOOKAHEAD_DAYS = 2  # suppress new overweight signals within this many days of an event


def get_upcoming_earnings_symbols(tickers: list[str] = TICKERS,
                                   lookahead_days: int = LOOKAHEAD_DAYS) -> set:
    """Symbols with an earnings release within the next lookahead_days."""
    today = date.today().isoformat()
    end = (date.today() + timedelta(days=lookahead_days)).isoformat()

    resp = requests.get(FINNHUB_EARNINGS_URL, params={
        "from": today, "to": end, "token": FINNHUB_API_KEY,
    }, timeout=30)
    resp.raise_for_status()
    entries = resp.json().get("earningsCalendar", [])
    return {e["symbol"] for e in entries if e.get("symbol") in tickers}


def get_pending_merger_symbols(tickers: list[str] = TICKERS) -> set:
    """Symbols involved in an announced, not-yet-completed merger.

    ca_sub_type filtering is based on inferred naming, not explicit Alpaca
    documentation (which doesn't spell out the distinction): MERGER_UPDATE
    appears to represent an active/pending announcement, MERGER_COMPLETION
    a finished one (confirmed empirically -- a completion record pulled
    live had null dates and no-op rates, consistent with "already done").
    Checks both initiating_symbol and target_symbol, since either side of
    a merger could be one of ours.
    """
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    today = date.today().isoformat()
    lookback = (date.today() - timedelta(days=90)).isoformat()  # API's own 90-day window limit

    request = GetCorporateAnnouncementsRequest(
        ca_types=[CorporateActionType.MERGER],
        since=lookback, until=today,
    )
    announcements = client.get_corporate_announcements(filter=request)
    pending = [a for a in announcements if a.ca_sub_type == CorporateActionSubType.MERGER_UPDATE]
    symbols = {a.initiating_symbol for a in pending} | {a.target_symbol for a in pending}
    return symbols & set(tickers)


def get_event_risk_symbols(tickers: list[str] = TICKERS) -> set:
    """Union of both event types -- the actual set rules.py should suppress
    new overweight signals for.
    """
    earnings = get_upcoming_earnings_symbols(tickers)
    mergers = get_pending_merger_symbols(tickers)
    return earnings | mergers


if __name__ == "__main__":
    earnings = get_upcoming_earnings_symbols()
    print(f"Upcoming earnings (next {LOOKAHEAD_DAYS} days): {earnings}")
    mergers = get_pending_merger_symbols()
    print(f"Pending mergers: {mergers}")

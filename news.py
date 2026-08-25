"""Fetch daily news sentiment (tone) per ticker from GDELT's DOC 2.0 API.

Bypasses the gdeltdoc package's timeline_search(), which raises
RateLimitError even on successful (HTTP 200) responses -- calls the raw
API directly instead. GDELT enforces roughly one request per 5 seconds,
AND rejects multi-year single-shot ranges (confirmed empirically: 1 month
and 1 year both work, a 5-year single query consistently 429s). Requests
are chunked by year and paced at 6s to stay under both limits.

Output aligns to (symbol, date) so it can be merged into the price/volume
feature table on the same keys.
"""
import time

import pandas as pd
import requests

from config import TICKER_TO_COMPANY

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_DELAY_SECONDS = 20  # bumped from 6->10->20 -- rate limiting proved persistent
RATE_LIMIT_BACKOFF_SECONDS = 30


def fetch_tone_timeline(company: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch one company's daily average tone for a single chunk (must be
    <= ~1 year; GDELT rejects wider single-shot ranges). Dates as
    'YYYY-MM-DD'. Returns columns: date (datetime64, tz-naive), tone (float).

    Retries once after a longer backoff on 429, since GDELT rate limiting
    has proven bursty rather than a clean fixed interval.
    """
    params = {
        "query": f'"{company}"',
        "mode": "timelinetone",
        "format": "json",
        "startdatetime": start_date.replace("-", "") + "000000",
        "enddatetime": end_date.replace("-", "") + "000000",
    }
    for attempt in (1, 2):
        resp = requests.get(GDELT_URL, params=params, timeout=30)
        if resp.status_code == 429 and attempt == 1:
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
            continue
        resp.raise_for_status()
        break

    points = resp.json()["timeline"][0]["data"]
    df = pd.DataFrame(points)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.rename(columns={"value": "tone"})[["date", "tone"]]


def _year_chunks(start_date: str, end_date: str):
    """Split a date range into <=1-year chunks as 'YYYY-MM-DD' string pairs."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []
    cur = start
    while cur < end:
        chunk_end = min(cur + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_all_sentiment(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily tone for every ticker in TICKER_TO_COMPANY, chunked by
    year and paced to respect GDELT's limits. Returns a DataFrame indexed
    by (symbol, date) with a single 'tone' column.
    """
    chunks = _year_chunks(start_date, end_date)
    frames = []
    first_call = True

    for ticker, company in TICKER_TO_COMPANY.items():
        ticker_frames = []
        for chunk_start, chunk_end in chunks:
            if not first_call:
                time.sleep(REQUEST_DELAY_SECONDS)
            first_call = False
            try:
                ticker_frames.append(fetch_tone_timeline(company, chunk_start, chunk_end))
            except Exception as e:
                print(f"  {ticker} ({company}) {chunk_start}..{chunk_end} FAILED: {e}")

        if ticker_frames:
            tone_df = pd.concat(ticker_frames)
            tone_df["symbol"] = ticker
            frames.append(tone_df)
            print(f"  {ticker} ({company}): {len(tone_df)} days")

    if not frames:
        raise RuntimeError(
            "Fetched zero tickers successfully -- likely still rate limited. "
            "Check the per-ticker FAILED messages above before retrying."
        )
    result = pd.concat(frames)
    return result.set_index(["symbol", "date"]).sort_index()


if __name__ == "__main__":
    sentiment = fetch_all_sentiment("2021-08-17", "2026-08-15")
    print(sentiment.shape)
    print(sentiment.head())
    sentiment.to_csv("sentiment_cache.csv")


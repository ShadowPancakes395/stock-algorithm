"""ONE-OFF comparison: score the same real headlines with VADER vs FinBERT
to see whether a finance-tuned transformer meaningfully disagrees with the
lexicon-based scorer we've been using. Not wired into any pipeline --
purely a same-day feasibility check before deciding whether the larger
backtesting-data-engineering effort (see chat) is worth it.
"""
from datetime import date, timedelta

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

from config import NEWSAPI_KEY, TICKER_TO_COMPANY

NEWSAPI_URL = "https://newsapi.org/v2/everything"
TEST_TICKERS = ["AAPL", "JPM", "XOM"]  # a small, varied sample -- tech, financial, energy

vader = SentimentIntensityAnalyzer()
print("Loading FinBERT (first run downloads ~400MB model weights)...")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")


def fetch_headlines(company: str, from_date: str, to_date: str, limit: int = 5) -> list:
    params = {
        "q": f'"{company}"', "from": from_date, "to": to_date,
        "sortBy": "relevancy", "pageSize": limit, "language": "en", "apiKey": NEWSAPI_KEY,
    }
    resp = requests.get(NEWSAPI_URL, params=params, timeout=30)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return [" ".join(filter(None, [a.get("title"), a.get("description")])) for a in articles]


if __name__ == "__main__":
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today = date.today().isoformat()

    for ticker in TEST_TICKERS:
        company = TICKER_TO_COMPANY[ticker]
        print(f"\n=== {ticker} ({company}) ===")
        headlines = fetch_headlines(company, yesterday, today)
        if not headlines:
            print("  No articles found.")
            continue

        for text in headlines:
            vader_score = vader.polarity_scores(text)["compound"]
            fb_result = finbert(text[:512])[0]  # FinBERT has a token limit
            fb_label, fb_score = fb_result["label"], fb_result["score"]
            print(f"  VADER: {vader_score:+.3f}  FinBERT: {fb_label:8s}({fb_score:.3f})  {text[:90]}")

"""Fetch and score TODAY's news sentiment per ticker, for live trading.

Distinct from news.py (GDELT, historical, backtest-only): this uses
NewsAPI, which only needs current-day coverage -- exactly the free-tier
limitation (articles up to ~1 month old) that made NewsAPI unusable for
backtesting but is a non-issue here.

Unlike GDELT, NewsAPI returns raw article text with no pre-computed tone
score, so sentiment is scored here with VADER (a lexicon-based scorer)
on each article's title + description, then averaged per ticker.

IMPORTANT CAVEAT (see chat for full discussion): the resulting sentiment
score is NOT on the same scale/distribution as the GDELT tone the model
was trained on. This is flagged, not silently papered over -- do not wire
this into trade.py until that mismatch is resolved.
"""
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import NEWSAPI_KEY, TICKER_TO_COMPANY

NEWSAPI_URL = "https://newsapi.org/v2/everything"
_analyzer = SentimentIntensityAnalyzer()


def fetch_ticker_sentiment(company: str, from_date: str, to_date: str) -> tuple[float, int]:
    """Fetch today's articles for one company and return (avg_compound_score,
    article_count). Returns (None, 0) if no articles found.
    """
    params = {
        "q": f'"{company}"',
        "from": from_date,
        "to": to_date,
        "sortBy": "relevancy",
        "pageSize": 100,
        "language": "en",
        "apiKey": NEWSAPI_KEY,
    }
    resp = requests.get(NEWSAPI_URL, params=params, timeout=30)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])

    if not articles:
        return None, 0

    scores = []
    for a in articles:
        text = " ".join(filter(None, [a.get("title"), a.get("description")]))
        if text.strip():
            scores.append(_analyzer.polarity_scores(text)["compound"])

    if not scores:
        return None, 0
    return sum(scores) / len(scores), len(scores)


def fetch_all_today_sentiment(from_date: str, to_date: str) -> dict:
    """Fetch and score today's sentiment for every ticker. Free tier allows
    100 requests/day; 28 tickers well within that, no pacing needed.
    Returns {ticker: sentiment_score or None}.
    """
    results = {}
    for ticker, company in TICKER_TO_COMPANY.items():
        try:
            score, count = fetch_ticker_sentiment(company, from_date, to_date)
        except Exception as e:
            print(f"  {ticker} ({company}) FAILED: {e}")
            results[ticker] = None
            continue
        results[ticker] = score
        print(f"  {ticker} ({company}): {count} articles, score={score}")
    return results


if __name__ == "__main__":
    from datetime import date, timedelta

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today = date.today().isoformat()

    sentiment = fetch_all_today_sentiment(yesterday, today)
    print()
    print(sentiment)

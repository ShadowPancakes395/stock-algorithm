"""Central config: tickers, thresholds, and API credentials."""
import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "JNJ", "PG",
    "XOM", "HD", "KO", "DIS", "V", "UNH", "COST", "WMT", "BAC", "PFE",
    "CSCO", "BA", "CAT", "HON", "AVGO", "AMD", "T", "VZ", "NEE", "LIN", "AMT",
]

# Classification thresholds on predicted up-probability
BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40

"""Central config: tickers, thresholds, and API credentials."""
import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "JPM", "JNJ", "PG",
    "XOM", "HD", "KO", "DIS", "V", "UNH", "COST", "WMT", "PFE",
    "CSCO", "BA", "CAT", "HON", "AVGO", "AMD", "T", "VZ", "NEE", "LIN", "AMT",
]

# Classification thresholds on predicted up-probability
BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40

# GDELT search terms per ticker. Some are deliberately specific to avoid
# false matches: "Caterpillar Inc" not "Caterpillar" (insect), "Visa Inc"
# not "Visa" (travel visas), "Advanced Micro Devices" not "AMD" (medical
# term), "AT&T" since the ticker "T" alone is unusable as a keyword.
TICKER_TO_COMPANY = {
    "AAPL": "Apple Inc", "MSFT": "Microsoft", "GOOGL": "Google",
    "NVDA": "Nvidia", "META": "Meta Platforms",
    "JPM": "JPMorgan Chase", "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble", "XOM": "ExxonMobil", "HD": "Home Depot",
    "KO": "Coca-Cola", "DIS": "Walt Disney", "V": "Visa Inc",
    "UNH": "UnitedHealth", "COST": "Costco", "WMT": "Walmart",
    "PFE": "Pfizer", "CSCO": "Cisco",
    "BA": "Boeing", "CAT": "Caterpillar Inc", "HON": "Honeywell",
    "AVGO": "Broadcom", "AMD": "Advanced Micro Devices", "T": "AT&T",
    "VZ": "Verizon", "NEE": "NextEra Energy", "LIN": "Linde plc",
    "AMT": "American Tower",
}

import os
import time
from sec_edgar_downloader import Downloader

# 1. SETUP: Define your identity (Required by SEC)
# Replace with your actual details
USER_AGENT = "Digvijay Singh (digvijay@arsaga.jp)" 
dl = Downloader("FinSage", USER_AGENT, "./data/raw")

# 2. TARGETS: The 30 companies from your Blueprint
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", # Tech
    "JPM", "GS", "BAC", "V", "MA",           # Finaßnce
    "JNJ", "PFE", "UNH", "ABBV", "MRK",      # Healthcare
    "WMT", "KO", "NKE", "MCD", "SBUX",       # Consumer
    "TSLA", "F", "GM", "RIVN", "LCID",       # EV
    "CRM", "SNOW", "PLTR", "NET", "DDOG"     # SaaS
]

def fetch_filings():
    for ticker in TICKERS:
        print(f"--- Downloading for {ticker} ---")
        
        # Download 10-Ks for the last 5 years
        dl.get("10-K", ticker, after="2020-01-01")
        
        # Download 10-Qs for the last 20 quarters
        dl.get("10-Q", ticker, after="2020-01-01")
        
        # SEC Limit: 10 requests per second. 
        # We add a small sleep to be safe and avoid 403 errors.
        time.sleep(0.2) 

if __name__ == "__main__":
    fetch_filings()
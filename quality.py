"""
Data source abstraction for OHLCV price data.

In production: swap simulate_ohlcv() for yf.download() or an internal API call.
The rest of the pipeline does not need to change.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def fetch(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for a list of tickers between start and end dates.

    Returns a DataFrame with columns:
        date, open, high, low, close, volume, ticker
    """
    print(f"[EXTRACT] Fetching {len(tickers)} tickers — {start} to {end}")
    frames = []
    for ticker in tickers:
        df = _fetch_single(ticker, start, end)
        df["ticker"] = ticker
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    print(f"[EXTRACT] Done — {len(raw):,} rows, {raw['ticker'].nunique()} tickers")
    return raw


def _fetch_single(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch data for a single ticker.
    Tries yfinance first, falls back to simulation if unavailable.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.reset_index().rename(columns={"Date": "date"})
        return df
    except ImportError:
        return simulate_ohlcv(ticker, start, end)


def simulate_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Simulate realistic OHLCV data using geometric Brownian motion.
    Used as a fallback when yfinance is not available.
    Same output structure as yf.download().
    """
    np.random.seed(abs(hash(ticker)) % 2**31)

    dates = pd.date_range(start=start, end=end, freq="B")
    n = len(dates)

    base_prices = {"AAPL": 180, "MSFT": 310, "GOOGL": 140, "AMZN": 170, "NVDA": 250}
    S0 = base_prices.get(ticker, 100)

    mu, sigma = 0.0003, 0.018
    returns = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.randn(n))
    close = S0 * np.cumprod(returns)

    daily_vol = np.abs(np.random.randn(n)) * 0.01
    high   = close * (1 + daily_vol)
    low    = close * (1 - daily_vol)
    open_  = close * (1 + np.random.randn(n) * 0.005)
    volume = np.random.randint(20_000_000, 80_000_000, n).astype(float)

    # Inject 1% missing values to simulate real data quality issues
    close[np.random.rand(n) < 0.01] = np.nan

    return pd.DataFrame({
        "date":   dates,
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    })

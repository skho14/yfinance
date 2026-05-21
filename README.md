"""
transform/indicators.py
========================
Financial indicator computation on cleaned OHLCV data.

Indicators:
    - log_return      Daily log return
    - ma_N            Simple moving average (configurable windows)
    - vol_20d         Annualised rolling volatility (20-day)
    - rsi_14          Relative Strength Index (14 periods)
    - drawdown        Percentage drawdown from rolling maximum
    - vwap            VWAP proxy ((H+L+C)/3)
    - trend_signal    Bullish / Bearish / Neutral (MA20 vs MA50 crossover)
"""

import pandas as pd
import numpy as np


def enrich(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add financial indicators to a cleaned OHLCV DataFrame.

    Args:
        df: Cleaned DataFrame from cleaner.py
        config: Optional dict with keys:
                  moving_averages: list of int windows (default [20, 50, 200])
                  rsi_period: int (default 14)
                  volatility_window: int (default 20)

    Returns:
        Enriched DataFrame with all indicators added as new columns.
    """
    cfg = config or {}
    ma_windows  = cfg.get("moving_averages", [20, 50, 200])
    rsi_period  = cfg.get("rsi_period", 14)
    vol_window  = cfg.get("volatility_window", 20)

    df = df.copy()
    g = df.groupby("ticker")["close"]

    # ── Log return ───────────────────────────────────────────────────────────
    df["log_return"] = g.transform(lambda x: np.log(x / x.shift(1)))

    # ── Moving averages ──────────────────────────────────────────────────────
    for w in ma_windows:
        df[f"ma_{w}"] = g.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())

    # ── Rolling volatility (annualised) ──────────────────────────────────────
    df[f"vol_{vol_window}d"] = g.transform(
        lambda x: x.pct_change()
                   .rolling(vol_window, min_periods=5)
                   .std() * np.sqrt(252)
    )

    # ── RSI ──────────────────────────────────────────────────────────────────
    df[f"rsi_{rsi_period}"] = g.transform(
        lambda x: _compute_rsi(x, rsi_period)
    )

    # ── Drawdown ─────────────────────────────────────────────────────────────
    df["drawdown"] = g.transform(
        lambda x: (x - x.cummax()) / x.cummax()
    )

    # ── VWAP proxy ───────────────────────────────────────────────────────────
    df["vwap"] = (df["close"] + df["high"] + df["low"]) / 3

    # ── Trend signal ─────────────────────────────────────────────────────────
    if "ma_20" in df.columns and "ma_50" in df.columns:
        df["trend_signal"] = np.where(
            df["ma_20"] > df["ma_50"], "bullish",
            np.where(df["ma_20"] < df["ma_50"], "bearish", "neutral")
        )

    cols = [c for c in df.columns if c not in ("date", "open", "high", "low", "close", "volume", "ticker", "is_outlier")]
    print(f"  [ENRICH] Indicators computed: {', '.join(cols)}")

    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using exponential weighted moving average (Wilder's method).

    Args:
        series: Close price series for a single ticker
        period: RSI lookback period (default 14)

    Returns:
        RSI series (values between 0 and 100)
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

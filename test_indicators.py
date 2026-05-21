"""
monitor/quality.py
===================
Data quality report generation for the financial ETL pipeline.

In production: push this report to a monitoring dashboard
(Grafana, DataDog, custom API endpoint).
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def report(df: pd.DataFrame, output_dir: str = "./output") -> pd.DataFrame:
    """
    Generate a per-ticker data quality and performance summary.

    Metrics per ticker:
        - Row count and date range
        - Missing values and outlier count
        - Average annualised volatility
        - Average RSI
        - Maximum drawdown
        - Total return over the period
        - % of bullish signal days

    Args:
        df: Enriched and cleaned DataFrame
        output_dir: Where to write quality_report.csv

    Returns:
        Summary DataFrame (one row per ticker)
    """
    print(f"\n[MONITOR] Generating data quality report...")

    rows = []
    for ticker, g in df.groupby("ticker"):
        vol_col = next((c for c in g.columns if c.startswith("vol_")), None)
        rsi_col = next((c for c in g.columns if c.startswith("rsi_")), None)

        row = {
            "ticker":           ticker,
            "n_rows":           len(g),
            "date_from":        g["date"].min().date(),
            "date_to":          g["date"].max().date(),
            "missing_close":    g["close"].isna().sum(),
            "n_outliers":       int(g.get("is_outlier", pd.Series([False]*len(g))).sum()),
            "avg_vol_pct":      round(g[vol_col].mean() * 100, 2) if vol_col else None,
            "avg_rsi":          round(g[rsi_col].mean(), 1) if rsi_col else None,
            "max_drawdown_pct": round(g["drawdown"].min() * 100, 2) if "drawdown" in g.columns else None,
            "total_return_pct": round((g["close"].iloc[-1] / g["close"].iloc[0] - 1) * 100, 2),
            "pct_bullish_days": round((g.get("trend_signal", pd.Series()) == "bullish").mean() * 100, 1)
                                 if "trend_signal" in g.columns else None,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    # ── Print report ─────────────────────────────────────────────────────────
    _print_report(summary, df)

    # ── Export ───────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "quality_report.csv")
    summary.to_csv(report_path, index=False)
    print(f"\n  [MONITOR] Report exported → {report_path}")

    return summary


def _print_report(summary: pd.DataFrame, df: pd.DataFrame) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  FINANCIAL DATASET — DATA QUALITY & SUMMARY REPORT")
    print(f"{sep}")
    print(f"  Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Tickers     : {', '.join(df['ticker'].unique())}")
    print(f"  Date range  : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total rows  : {len(df):,}")
    print(sep)

    for _, r in summary.iterrows():
        print(f"\n  [{r['ticker']}]")
        print(f"    Rows          : {r['n_rows']:,}")
        print(f"    Missing close : {r['missing_close']}")
        print(f"    Outliers      : {r['n_outliers']}")
        if r["avg_vol_pct"]:
            print(f"    Avg volatility: {r['avg_vol_pct']}% (annualised)")
        if r["avg_rsi"]:
            print(f"    Avg RSI       : {r['avg_rsi']}")
        if r["max_drawdown_pct"]:
            print(f"    Max drawdown  : {r['max_drawdown_pct']}%")
        print(f"    Total return  : {r['total_return_pct']}%")
        if r["pct_bullish_days"]:
            print(f"    Bullish days  : {r['pct_bullish_days']}%")

    print(f"\n{sep}")

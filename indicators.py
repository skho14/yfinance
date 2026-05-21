"""
load/exporter.py
=================
Export enriched financial dataset to Parquet (or CSV fallback).

Design:
    - Full dataset exported as a single file
    - Per-ticker partitions for efficient downstream querying
    - Parquet preferred (columnar, compressed, fast for analytics)
    - CSV fallback if Parquet engine is unavailable
"""

import pandas as pd
import os


def export(df: pd.DataFrame, config: dict = None) -> str:
    """
    Export enriched DataFrame to disk.

    Args:
        df: Enriched DataFrame from indicators.py
        config: Optional dict with keys:
                  dir: output directory (default "./output")
                  format: "parquet" | "csv" (default "parquet")
                  partition_by_ticker: bool (default True)

    Returns:
        Path of the main output file.
    """
    cfg            = config or {}
    output_dir     = cfg.get("dir", "./output")
    fmt            = cfg.get("format", "parquet")
    partition      = cfg.get("partition_by_ticker", True)

    os.makedirs(output_dir, exist_ok=True)

    # ── Full dataset ─────────────────────────────────────────────────────────
    main_path = _write(df, os.path.join(output_dir, "financial_dataset"), fmt)
    print(f"  [LOAD] Full dataset → {main_path}")

    # ── Per-ticker partitions ────────────────────────────────────────────────
    if partition:
        for ticker, group in df.groupby("ticker"):
            path = _write(group, os.path.join(output_dir, ticker), fmt)
        print(f"  [LOAD] Ticker partitions → {output_dir}/<TICKER>.{fmt}")

    return main_path


def _write(df: pd.DataFrame, base_path: str, fmt: str) -> str:
    """Write DataFrame to disk. Falls back to CSV if Parquet engine unavailable."""
    if fmt == "parquet":
        path = base_path + ".parquet"
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            fmt = "csv"

    path = base_path + ".csv"
    df.to_csv(path, index=False)
    return path

"""
Financial ETL Pipeline — entry point.

Usage:
    python main.py

Config:
    Edit config/config.yaml to change tickers, date range, and output settings.
"""

import yaml
from datetime import datetime

from src.extract.fetcher    import fetch
from src.transform.cleaner  import clean
from src.transform.indicators import enrich
from src.load.exporter      import export
from src.monitor.quality    import report


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run():
    print("=" * 68)
    print("  FINANCIAL ETL PIPELINE")
    print("=" * 68 + "\n")

    cfg = load_config()
    p   = cfg["pipeline"]

    t0 = datetime.now()

    # EXTRACT 
    raw = fetch(
        tickers=p["tickers"],
        start=p["date_range"]["start"],
        end=p["date_range"]["end"],
    )

    # TRANSFORM 
    print("\n[TRANSFORM] Cleaning...")
    cleaned = clean(raw, config=p["quality"])

    print("\n[TRANSFORM] Enriching with indicators...")
    enriched = enrich(cleaned, config=p["indicators"])

    # LOAD
    print("\n[LOAD] Exporting...")
    path = export(enriched, config=p["output"])

    # MONITOR
    summary = report(enriched, output_dir=p["output"]["dir"])

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n Pipeline completed in {elapsed:.2f}s")
    print(f"    Rows processed : {len(enriched):,}")
    print(f"    Columns        : {list(enriched.columns)}")
    print(f"    Output         : {p['output']['dir']}/")


if __name__ == "__main__":
    run()

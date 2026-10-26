"""Data quality checks. Exits with code 1 if any check fails."""
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd

MIN_MW, MAX_MW = 15_000, 100_000  # sane bounds for ERCOT hourly demand


def check(df: pd.DataFrame) -> list[str]:
    errors = []
    if df.empty:
        return ["dataset is empty"]

    # 1. Uniqueness: each hour should appear once
    if df["timestamp"].duplicated().any():
        errors.append("duplicate timestamps")

    # 2. Completeness: almost no missing values
    missing = df["demand_mw"].isna().mean()
    if missing > 0.01:
        errors.append(f"{missing:.1%} of demand values are missing (limit 1%)")

    # 3. Validity: values inside a realistic range
    out_of_range = ((df["demand_mw"] < MIN_MW) | (df["demand_mw"] > MAX_MW)).sum()
    if out_of_range > 5:
        errors.append(f"{out_of_range} values outside {MIN_MW:,}-{MAX_MW:,} MW")

    # 4. Continuity: few gaps in the hourly series
    gaps = (df["timestamp"].sort_values().diff() > pd.Timedelta(hours=1)).sum()
    if gaps > 24:
        errors.append(f"{gaps} gaps in the hourly series (limit 24)")

    # 5. Freshness: the latest data is recent
    latest = df["timestamp"].max()
    if latest < datetime.now(UTC) - timedelta(hours=72):
        errors.append(f"data is stale: latest hour is {latest}")

    return errors


if __name__ == "__main__":
    problems = check(pd.read_parquet("data/demand.parquet"))
    for p in problems:
        print(f"FAIL: {p}")
    if problems:
        sys.exit(1)
    print("All data checks passed")

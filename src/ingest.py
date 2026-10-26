"""Download hourly ERCOT demand from the EIA API into data/demand.parquet."""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

DATA_PATH = Path("data/demand.parquet")
URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
PAGE = 5000  # the API returns at most 5,000 rows per call


def fetch(start: datetime, end: datetime) -> pd.DataFrame:
    params = {
        "api_key": os.environ["EIA_API_KEY"],
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "ERCO",
        "facets[type][]": "D",
        "start": start.strftime("%Y-%m-%dT%H"),
        "end": end.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": PAGE,
    }
    rows, offset = [], 0
    while True:  # page through results
        params["offset"] = offset
        resp = requests.get(URL, params=params, timeout=60)
        resp.raise_for_status()
        batch = resp.json()["response"]["data"]
        rows.extend(batch)
        print(f"  fetched {len(rows):,} rows...")
        if len(batch) < PAGE:
            break
        offset += PAGE

    df = pd.DataFrame(rows).rename(columns={"period": "timestamp", "value": "demand_mw"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H", utc=True)
    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")
    return df[["timestamp", "demand_mw"]]


def main() -> None:
    end = datetime.now(timezone.utc)
    if DATA_PATH.exists():
        old = pd.read_parquet(DATA_PATH)
        # Re-pull the last 3 days: recent hours can arrive late or be corrected.
        start = old["timestamp"].max().to_pydatetime() - timedelta(days=3)
    else:
        old = None
        start = end - timedelta(days=3 * 365)

    print(f"Fetching {start:%Y-%m-%d %H:00} to {end:%Y-%m-%d %H:00} UTC")
    new = fetch(start, end)
    df = new if old is None else pd.concat([old, new])
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp")

    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_parquet(DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows; latest hour {df['timestamp'].max()}")


if __name__ == "__main__":
    main()

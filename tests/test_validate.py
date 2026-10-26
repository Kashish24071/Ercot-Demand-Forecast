from datetime import UTC, datetime

import pandas as pd

from validate import check


def recent(hours=200):
    """Fake clean data ending at the current hour."""
    end = pd.Timestamp(datetime.now(UTC)).floor("h")
    ts = pd.date_range(end=end, periods=hours, freq="h")
    return pd.DataFrame({"timestamp": ts, "demand_mw": 45_000.0})


def test_clean_data_passes():
    assert check(recent()) == []


def test_duplicates_fail():
    df = recent()
    df = pd.concat([df, df.tail(1)])
    assert any("duplicate" in e for e in check(df))


def test_out_of_range_fails():
    df = recent()
    df.loc[:10, "demand_mw"] = 0
    assert any("outside" in e for e in check(df))

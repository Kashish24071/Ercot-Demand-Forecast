import pandas as pd

from features import FEATURES, future_frame, make_features


def sample(hours=400):
    """Fake data: 400 hours of steadily rising demand."""
    ts = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "demand_mw": range(40_000, 40_000 + hours)})


def test_all_features_present():
    out = make_features(sample())
    assert set(FEATURES) <= set(out.columns)


def test_lag_24_is_value_from_yesterday():
    out = make_features(sample())
    assert out["lag_24"].iloc[30] == out["demand_mw"].iloc[6]


def test_future_frame_has_no_missing_features():
    future = future_frame(sample(), hours=24)
    assert len(future) == 25
    assert not future[FEATURES].isna().any().any()

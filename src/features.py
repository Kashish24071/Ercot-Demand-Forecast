"""Feature engineering shared by training and the API."""
import pandas as pd

FEATURES = ["hour", "dayofweek", "month", "is_weekend",
            "lag_24", "lag_48", "lag_168", "roll_24_mean"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    s = df.set_index("timestamp")["demand_mw"].asfreq("h")
    s = s.interpolate(limit=3, limit_area="inside")  # fill short gaps only
    out = pd.DataFrame({"demand_mw": s})
    local = out.index.tz_convert("America/Chicago")  # ERCOT runs on Central time
    out["hour"] = local.hour
    out["dayofweek"] = local.dayofweek
    out["month"] = local.month
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    out["lag_24"] = s.shift(24)    # same hour yesterday
    out["lag_48"] = s.shift(48)
    out["lag_168"] = s.shift(168)  # same hour last week
    out["roll_24_mean"] = s.shift(24).rolling(24).mean()
    return out


def future_frame(df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """Feature rows for the next `hours` hours after the last observed hour."""
    last = df["timestamp"].max()
    future = pd.DataFrame({
        "timestamp": pd.date_range(last + pd.Timedelta(hours=1), periods=hours, freq="h"),
        "demand_mw": float("nan"),
    })
    return make_features(pd.concat([df, future])).tail(hours)

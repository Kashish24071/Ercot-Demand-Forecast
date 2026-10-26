"""Forecast API: serves the latest model's 24-hour forecast."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from features import FEATURES, future_frame

app = FastAPI(title="ERCOT Demand Forecast API")
model = joblib.load(ROOT / "models/model.joblib")
history = pd.read_parquet(ROOT / "data/demand.parquet")
# "High-stress" = forecast at or above the 95th percentile of the past year
STRESS_MW = history.tail(24 * 365)["demand_mw"].quantile(0.95)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast")
def forecast(hours: int = 24):
    hours = max(1, min(hours, 24))
    future = future_frame(history, hours)
    preds = model.predict(future[FEATURES])
    return [
        {"timestamp": ts.isoformat(), "forecast_mw": round(float(p)),
         "high_stress": bool(p >= STRESS_MW)}
        for ts, p in zip(future.index, preds)
    ]


@app.get("/history")
def recent(days: int = 7):
    df = history.tail(24 * max(1, min(days, 60))).dropna()
    return [{"timestamp": t.isoformat(), "demand_mw": v}
            for t, v in zip(df["timestamp"], df["demand_mw"])]


@app.get("/metrics")
def metrics():
    return json.loads((ROOT / "models/metrics.json").read_text())

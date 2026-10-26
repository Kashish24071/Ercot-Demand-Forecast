"""Streamlit dashboard: shows actual demand, the 24-hour forecast and model accuracy."""
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="ERCOT Demand Forecast", layout="wide")


def api_url() -> str:
    try:
        return st.secrets["API_URL"]  # set in Streamlit Cloud
    except (FileNotFoundError, KeyError):  # no secrets locally
        return os.getenv("API_URL", "http://localhost:8000")  # local fallback


API = api_url()


@st.cache_data(ttl=600)  # re-fetch at most every 10 minutes
def get(path: str):
    r = requests.get(f"{API}{path}", timeout=90)  # long timeout: the API may be waking up
    r.raise_for_status()
    return r.json()


st.title("ERCOT electricity demand: next 24 hours")
st.caption("Hourly demand for the Texas grid, forecast by a model that retrains itself every day.")

with st.spinner("Loading forecast (the API can take up to a minute to wake up)..."):
    hist = pd.DataFrame(get("/history?days=7"))
    fc = pd.DataFrame(get("/forecast"))
    m = get("/metrics")

# Headline numbers
c1, c2, c3 = st.columns(3)
c1.metric("Model error (MAPE, last 14 days)", f"{m['mape']:.1%}")
c2.metric("Naive baseline MAPE", f"{m['baseline_mape']:.1%}")
c3.metric("Peak forecast, next 24 h", f"{fc['forecast_mw'].max():,.0f} MW")

# Actual vs forecast chart
chart = pd.concat([
    hist.rename(columns={"demand_mw": "mw"}).assign(series="Actual"),
    fc.rename(columns={"forecast_mw": "mw"}).assign(series="Forecast"),
])
chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True).dt.tz_convert("America/Chicago")
fig = px.line(chart, x="timestamp", y="mw", color="series",
              labels={"mw": "Demand (MW)", "timestamp": "Central time", "series": ""})
st.plotly_chart(fig)

# High-stress hours
st.subheader("High-stress hours ahead")
st.caption("Hours forecast at or above the top 5% of the past year's demand.")
stress = fc[fc["high_stress"]].copy()
if stress.empty:
    st.success("No high-stress hours in the next 24 hours.")
else:
    stress["Central time"] = (pd.to_datetime(stress["timestamp"], utc=True)
                              .dt.tz_convert("America/Chicago").dt.strftime("%a %b %d, %I %p"))
    stress["Forecast (MW)"] = stress["forecast_mw"].map("{:,}".format)
    st.dataframe(stress[["Central time", "Forecast (MW)"]], hide_index=True)

st.caption(f"Model trained {m['trained_at']} UTC on data through {m['data_through']}.")

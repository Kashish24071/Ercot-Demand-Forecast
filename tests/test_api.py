from pathlib import Path

import pytest

if not Path("models/model.joblib").exists():
    pytest.skip("no trained model yet", allow_module_level=True)

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_forecast_shape():
    body = client.get("/forecast").json()
    assert len(body) == 24
    assert all(15_000 < row["forecast_mw"] < 100_000 for row in body)

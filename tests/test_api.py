"""Test suite for FastAPI endpoints of Edge Defect Inspection & SPC Monitor."""

import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_api_inspect_synthetic_scratch(client):
    payload = {"defect_type": "scratch", "seed": 42}
    res = client.post("/inspect/synthetic", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["has_defect"] is True
    assert data["fft_spectral_energy"] > 0.0
    assert len(data["bounding_boxes"]) >= 1
    assert "circularity" in data["bounding_boxes"][0]


def test_api_spc_status_endpoint(client):
    res = client.get("/spc/status")
    assert res.status_code == 200
    data = res.json()
    assert "ucl_3sigma" in data
    assert "first_pass_yield_pct" in data


def test_api_benchmark_endpoint(client):
    res = client.get("/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert data["frames_evaluated"] == 30
    assert data["throughput_fps"] > 10.0

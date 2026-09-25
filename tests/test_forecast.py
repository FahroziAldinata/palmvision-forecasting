import re

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

TEST_BLOK_UUID = "c1a2b3c4-d5e6-7f8a-9b0c-1d2e3f4a5b6c"


def test_forecast_without_api_key_returns_401():
    response = client.post(
        "/api/v1/forecast",
        json={"blok_id": TEST_BLOK_UUID, "horizon_bulan": 3},
    )
    assert response.status_code == 401


def test_forecast_with_invalid_api_key_returns_401():
    response = client.post(
        "/api/v1/forecast",
        headers={"X-API-Key": "wrong-key"},
        json={"blok_id": TEST_BLOK_UUID, "horizon_bulan": 3},
    )
    assert response.status_code == 401


def test_forecast_with_valid_contract_returns_200():
    payload = {
        "blok_id": TEST_BLOK_UUID,
        "horizon_bulan": 3,
        "fitur_tambahan": {
            "curah_hujan_historis": [120.5, 140.2, 160.0],
            "usia_tanaman_bulan": 84,
        },
    }
    response = client.post(
        "/api/v1/forecast",
        headers={"X-API-Key": "palmvision-dev-key"},
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["blok_id"] == TEST_BLOK_UUID
    assert len(data["proyeksi"]) == 3

    # Verifikasi format periode YYYY-MM
    assert re.match(r"^\d{4}-\d{2}$", data["proyeksi"][0]["periode"])
    assert data["proyeksi"][0]["nilai_kg"] >= 0.0
    assert data["proyeksi"][0]["interval_bawah"] <= data["proyeksi"][0]["interval_atas"]
    assert 0.0 <= data["mape_model"] < 1.0
    assert data["versi_model"].startswith("prophet-v")


def test_train_endpoint_with_historical_records():
    records = [
        {"ds": f"2025-{m:02d}-01", "y": 15000.0 + (m * 200), "curah_hujan": 150.0, "usia_tanaman_bulan": 70 + m}
        for m in range(1, 13)
    ]
    payload = {
        "blok_id": TEST_BLOK_UUID,
        "records": records,
        "versi_model": "prophet-v1-test",
    }
    response = client.post(
        "/api/v1/train",
        headers={"X-API-Key": "palmvision-dev-key"},
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["blok_id"] == TEST_BLOK_UUID
    assert data["versi_model"] == "prophet-v1-test"
    assert data["records_count"] == 12
    assert 0.0 <= data["mape_model"] < 1.0
    assert data["status"] == "trained"


def test_backtest_endpoint_returns_valid_mape():
    records = [
        {"ds": f"2025-{m:02d}-01", "y": 18000.0 + (m * 150), "curah_hujan": 120.0, "usia_tanaman_bulan": 60 + m}
        for m in range(1, 13)
    ]
    payload = {
        "blok_id": TEST_BLOK_UUID,
        "records": records,
        "holdout_months": 3,
    }
    response = client.post(
        "/api/v1/backtest",
        headers={"X-API-Key": "palmvision-dev-key"},
        json=payload,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["blok_id"] == TEST_BLOK_UUID
    assert 0.0 <= data["mape_model"] < 1.0
    assert len(data["comparisons"]) == 3
    assert data["comparisons"][0]["y_actual"] > 0
    assert data["comparisons"][0]["y_pred"] > 0

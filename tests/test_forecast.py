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
            "curah_hujan_historis": [120.5, 140.2],
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
    assert len(data["proyeksi"]) > 0
    assert data["proyeksi"][0]["periode"] == "2026-11"
    assert data["proyeksi"][0]["nilai_kg"] == 18420.0
    assert data["mape_model"] == 0.087
    assert data["versi_model"] == "prophet-v3-2026Q3"

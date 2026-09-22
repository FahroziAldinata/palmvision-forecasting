from fastapi import APIRouter

from app.schemas.forecast_request import ForecastRequest, ForecastResponse, ProyeksiItem

router = APIRouter()


@router.post("/forecast", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest) -> ForecastResponse:
    """
    Endpoint stub forecasting Phase 0.
    Hanya memvalidasi skema request/response sesuai kontrak PRD 9.3
    dan mengembalikan data dummy statis tanpa machine learning/training.
    """
    dummy_proyeksi = [
        ProyeksiItem(
            periode="2026-11",
            nilai_kg=18420.0,
            interval_bawah=16100.0,
            interval_atas=20700.0,
        )
    ]

    return ForecastResponse(
        blok_id=request.blok_id,
        proyeksi=dummy_proyeksi,
        mape_model=0.087,
        versi_model="prophet-v3-2026Q3",
    )

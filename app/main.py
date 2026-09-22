import os

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from app.api.v1.routes_forecast import router as forecast_router

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

EXPECTED_API_KEY = os.getenv("API_KEY") or os.getenv("FORECASTING_API_KEY") or "palmvision-dev-key"

app = FastAPI(
    title="PalmVision Forecasting Service",
    description="Layanan terpisah time-series forecasting kelapa sawit untuk PalmVision",
    version="0.1.0",
)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key or api_key != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "palmvision-forecasting",
    }


app.include_router(
    forecast_router,
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)],
    tags=["Forecast"],
)

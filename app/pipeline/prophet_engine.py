import logging
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

from app.schemas.forecast_request import (
    BacktestComparisonItem,
    HistoricalRecord,
    ProyeksiItem,
)

logger = logging.getLogger("palmvision.forecasting")


def create_prophet_model(
    n_points: int = 12,
    has_rainfall: bool = True,
    has_age: bool = True,
) -> Prophet:
    """
    Inisialisasi model Prophet dengan konfigurasi khusus agrikultur kelapa sawit:
    - Yearly seasonality aktif adaptif (bila data >= 12 bulan)
    - Weekly & daily seasonality dimatikan (agregasi bulanan)
    - Mode aditif untuk kestabilan numerik
    - Interval kepercayaan 80% (interval_width=0.80)
    """
    yearly: bool | int = False
    if n_points >= 24:
        yearly = 10
    elif n_points >= 12:
        yearly = 3

    model = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.80,
    )

    if has_rainfall:
        model.add_regressor("curah_hujan", standardize=True)
    if has_age:
        model.add_regressor("usia_tanaman_bulan", standardize=True)

    return model


def prepare_dataframe(records: list[HistoricalRecord]) -> pd.DataFrame:
    """
    Mengubah list HistoricalRecord menjadi DataFrame standar Prophet ('ds', 'y', regressor).
    """
    data = []
    for r in records:
        ds_str = r.ds
        # Format periode bulanan YYYY-MM menjadi awal bulan YYYY-MM-01
        if len(ds_str) == 7:
            ds_str = f"{ds_str}-01"

        data.append({
            "ds": pd.to_datetime(ds_str),
            "y": float(r.y),
            "curah_hujan": float(r.curah_hujan or 0.0),
            "usia_tanaman_bulan": float(r.usia_tanaman_bulan or 0),
        })

    df = pd.DataFrame(data)
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def backtest_model(
    records: list[HistoricalRecord],
    holdout_months: int = 3,
) -> tuple[float, list[BacktestComparisonItem]]:
    """
    Menjalankan prosedur backtesting:
    Melatih model dengan data sampai periode (T - holdout), lalu menguji
    prediksi terhadap periode holdout aktual yang sudah diketahui.
    Mengembalikan nilai MAPE dan daftar perbandingan aktual vs prediksi.
    """
    df = prepare_dataframe(records)
    n = len(df)

    if n <= holdout_months:
        holdout_months = max(1, n // 3)

    train_df = df.iloc[:-holdout_months].copy()
    test_df = df.iloc[-holdout_months:].copy()

    # Validasi keberadaan regressor pada train set
    has_rainfall = bool(train_df["curah_hujan"].nunique() > 1)
    has_age = bool(train_df["usia_tanaman_bulan"].nunique() > 1)

    model = create_prophet_model(n_points=len(train_df), has_rainfall=has_rainfall, has_age=has_age)
    model.fit(train_df)

    # Buat future dataframe sesuai periode test
    future = test_df[["ds", "curah_hujan", "usia_tanaman_bulan"]].copy()
    forecast = model.predict(future)

    comparisons: list[BacktestComparisonItem] = []
    apes = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        y_true = float(row["y"])
        y_pred = float(max(0.0, forecast.iloc[i]["yhat"]))
        error_abs = abs(y_true - y_pred)
        ape = error_abs / max(y_true, 1.0)
        apes.append(ape)

        periode_str = row["ds"].strftime("%Y-%m")
        comparisons.append(
            BacktestComparisonItem(
                periode=periode_str,
                y_actual=round(y_true, 2),
                y_pred=round(y_pred, 2),
                error_abs=round(error_abs, 2),
                ape=round(ape, 4),
            )
        )

    mape = float(np.mean(apes)) if apes else 0.08
    return mape, comparisons


def train_and_fit_prophet(records: list[HistoricalRecord]) -> tuple[Prophet, float, dict[str, Any]]:
    """
    Melatih model Prophet penuh pada seluruh data historis dan menghitung skor backtest MAPE.
    """
    df = prepare_dataframe(records)

    # 1. Hitung backtesting MAPE
    mape, _ = backtest_model(records, holdout_months=min(3, max(1, len(records) // 4)))

    # 2. Train pada seluruh dataset
    has_rainfall = bool(df["curah_hujan"].nunique() > 1)
    has_age = bool(df["usia_tanaman_bulan"].nunique() > 1)

    model = create_prophet_model(n_points=len(df), has_rainfall=has_rainfall, has_age=has_age)
    model.fit(df)

    metadata = {
        "records_count": len(df),
        "start_ds": df["ds"].min().strftime("%Y-%m-%d"),
        "end_ds": df["ds"].max().strftime("%Y-%m-%d"),
        "has_rainfall": has_rainfall,
        "has_age": has_age,
        "mean_y": float(df["y"].mean()),
        "last_age": float(df["usia_tanaman_bulan"].iloc[-1]),
        "mean_rainfall": float(df["curah_hujan"].mean()),
    }

    return model, mape, metadata


def generate_prophet_forecast(
    model: Prophet,
    horizon_months: int,
    metadata: dict[str, Any] | None = None,
    fitur_tambahan: Any = None,
) -> list[ProyeksiItem]:
    """
    Melakukan inferensi time-series ke depan sejumlah horizon_months.
    """
    future = model.make_future_dataframe(periods=horizon_months, freq="MS")
    future_only = future.tail(horizon_months).copy()

    # Ekstraksi regressor masa depan
    meta = metadata or {}
    last_age = meta.get("last_age", 84.0)
    mean_rainfall = meta.get("mean_rainfall", 150.0)

    # Curah hujan dari fitur_tambahan jika tersedia
    curah_hujan_inputs = getattr(fitur_tambahan, "curah_hujan_historis", None) or []
    usia_input = getattr(fitur_tambahan, "usia_tanaman_bulan", None)

    future_rainfall = []
    future_age = []

    for i in range(horizon_months):
        # Usia tanaman bertambah 1 bulan tiap langkah
        age_val = float(usia_input + i) if usia_input is not None else float(last_age + i + 1)
        rain_val = float(curah_hujan_inputs[i]) if i < len(curah_hujan_inputs) else mean_rainfall

        future_rainfall.append(rain_val)
        future_age.append(age_val)

    if "curah_hujan" in model.extra_regressors:
        future_only["curah_hujan"] = future_rainfall
    if "usia_tanaman_bulan" in model.extra_regressors:
        future_only["usia_tanaman_bulan"] = future_age

    forecast = model.predict(future_only)

    proyeksi: list[ProyeksiItem] = []
    for _, row in forecast.iterrows():
        periode_str = row["ds"].strftime("%Y-%m")
        # Produksi kelapa sawit tidak mungkin negatif
        val = float(max(0.0, row["yhat"]))
        lower = float(max(0.0, row["yhat_lower"]))
        upper = float(max(val, row["yhat_upper"]))

        proyeksi.append(
            ProyeksiItem(
                periode=periode_str,
                nilai_kg=round(val, 2),
                interval_bawah=round(lower, 2),
                interval_atas=round(upper, 2),
            )
        )

    return proyeksi

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class FiturTambahan(BaseModel):
    curah_hujan_historis: list[float] | None = Field(
        default=None,
        description="Curah hujan historis dalam satuan mm",
    )
    usia_tanaman_bulan: int | None = Field(
        default=None,
        ge=0,
        description="Usia tanaman dalam satuan bulan",
    )


class ForecastRequest(BaseModel):
    blok_id: UUID = Field(..., description="UUID dari blok perkebunan")
    horizon_bulan: int = Field(
        ...,
        ge=1,
        le=12,
        description="Jumlah bulan proyeksi ke depan (1-12 bulan)",
    )
    fitur_tambahan: FiturTambahan | None = Field(
        default=None,
        description="Fitur agronomis tambahan pendukung proyeksi",
    )


class ProyeksiItem(BaseModel):
    periode: str = Field(..., description="Periode proyeksi dalam format YYYY-MM")
    nilai_kg: float = Field(..., description="Nilai proyeksi produksi dalam satuan kg")
    interval_bawah: float = Field(..., description="Batas bawah interval kepercayaan dalam kg")
    interval_atas: float = Field(..., description="Batas atas interval kepercayaan dalam kg")


class ForecastResponse(BaseModel):
    blok_id: UUID = Field(..., description="UUID dari blok perkebunan")
    proyeksi: list[ProyeksiItem] = Field(..., description="Daftar item proyeksi bulanan")
    mape_model: float = Field(..., description="Mean Absolute Percentage Error model")
    versi_model: str = Field(..., description="Identifier versi model forecasting")


class HistoricalRecord(BaseModel):
    ds: str = Field(..., description="Tanggal awal bulan periode histori (YYYY-MM-DD atau YYYY-MM)")
    y: float = Field(..., ge=0, description="Total produksi aktual kg dalam bulan tersebut")
    curah_hujan: float | None = Field(default=0.0, description="Total curah hujan bulanan dalam mm")
    usia_tanaman_bulan: int | None = Field(default=0, ge=0, description="Usia tanaman dalam bulan")


class TrainRequest(BaseModel):
    blok_id: UUID = Field(..., description="UUID blok yang akan dilatih")
    records: list[HistoricalRecord] = Field(..., min_length=6, description="Data historis produksi bulanan (min 6 bulan)")
    versi_model: str | None = Field(default=None, description="Nama versi kustom (opsional)")


class TrainResponse(BaseModel):
    blok_id: UUID = Field(..., description="UUID blok")
    versi_model: str = Field(..., description="Versi model tersimpan di registry")
    mape_model: float = Field(..., description="Hasil backtesting MAPE model")
    records_count: int = Field(..., description="Jumlah data poin yang digunakan dalam training")
    status: str = Field(default="trained", description="Status pelatihan model")


class BacktestRequest(BaseModel):
    blok_id: UUID = Field(..., description="UUID blok")
    records: list[HistoricalRecord] = Field(..., min_length=6, description="Data historis untuk backtesting")
    holdout_months: int = Field(default=3, ge=1, le=6, description="Jumlah bulan terakhir sebagai periode holdout")


class BacktestComparisonItem(BaseModel):
    periode: str = Field(..., description="Bulan periode holdout")
    y_actual: float = Field(..., description="Nilai aktual produksi (kg)")
    y_pred: float = Field(..., description="Nilai prediksi model (kg)")
    error_abs: float = Field(..., description="Selisih absolut |actual - pred|")
    ape: float = Field(..., description="Absolute Percentage Error")


class BacktestResponse(BaseModel):
    blok_id: UUID = Field(..., description="UUID blok")
    mape_model: float = Field(..., description="Rata-rata MAPE pada periode holdout")
    comparisons: list[BacktestComparisonItem] = Field(default_factory=list, description="Detail perbandingan per bulan")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata model dan evaluasi")

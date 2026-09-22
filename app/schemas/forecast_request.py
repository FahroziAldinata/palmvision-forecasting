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

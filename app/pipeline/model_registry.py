import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import joblib
from prophet import Prophet

logger = logging.getLogger("palmvision.forecasting")

BASE_REGISTRY_DIR = Path(__file__).resolve().parent.parent / "storage" / "model_registry"


def get_block_registry_dir(blok_id: str | UUID) -> Path:
    block_dir = BASE_REGISTRY_DIR / str(blok_id)
    block_dir.mkdir(parents=True, exist_ok=True)
    return block_dir


def generate_version_name(blok_id: str | UUID) -> str:
    """
    Format versi model sesuai spesifikasi PRD 9.3:
    prophet-v{n}-{periode} (contoh: prophet-v1-2026Q4)
    """
    block_dir = get_block_registry_dir(blok_id)
    existing_models = list(block_dir.glob("model_*.joblib"))
    next_v = len(existing_models) + 1

    now = datetime.now(UTC)
    quarter = (now.month - 1) // 3 + 1
    period_str = f"{now.year}Q{quarter}"

    return f"prophet-v{next_v}-{period_str}"


def save_model(
    blok_id: str | UUID,
    model: Prophet,
    mape: float,
    metadata: dict[str, Any] | None = None,
    custom_version: str | None = None,
) -> str:
    """
    Menyimpan model terlatih dan file metadata ke storage registry per blok.
    """
    block_dir = get_block_registry_dir(blok_id)
    version = custom_version or generate_version_name(blok_id)

    model_path = block_dir / f"model_{version}.joblib"
    meta_path = block_dir / f"meta_{version}.json"

    joblib.dump(model, model_path)

    full_meta = {
        "blok_id": str(blok_id),
        "versi_model": version,
        "mape_model": round(mape, 4),
        "saved_at": datetime.now(UTC).isoformat(),
        **(metadata or {}),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)

    # Simpan pointer versi aktif terbaru
    latest_pointer = block_dir / "latest.json"
    with open(latest_pointer, "w", encoding="utf-8") as f:
        json.dump({"latest_version": version}, f)

    logger.info("Model %s untuk blok %s berhasil disimpan di registry", version, blok_id)
    return version


def load_latest_model(blok_id: str | UUID) -> tuple[Prophet | None, dict[str, Any] | None]:
    """
    Memuat model aktif terbaru untuk blok_id dari registry.
    """
    block_dir = get_block_registry_dir(blok_id)
    latest_pointer = block_dir / "latest.json"

    if not latest_pointer.exists():
        return None, None

    try:
        with open(latest_pointer, "r", encoding="utf-8") as f:
            pointer = json.load(f)
        version = pointer.get("latest_version")
        if not version:
            return None, None

        model_path = block_dir / f"model_{version}.joblib"
        meta_path = block_dir / f"meta_{version}.json"

        if not model_path.exists() or not meta_path.exists():
            return None, None

        model = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return model, meta
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.error("Gagal memuat model untuk blok %s: %s", blok_id, e)
        return None, None

from __future__ import annotations

from pathlib import Path
import re

from fastapi import HTTPException


ALLOWED_UPLOAD_SUFFIXES = {
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".png",
    ".jpg",
    ".jpeg",
}


def safe_filename(filename: str | None) -> str:
    original = filename or "uploaded_slide"
    original = Path(original).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", original)
    suffix = Path(cleaned).suffix.lower()

    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                f"Allowed extensions: {sorted(ALLOWED_UPLOAD_SUFFIXES)}"
            ),
        )

    return cleaned

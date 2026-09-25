from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

try:
    import openslide
except ImportError as exc:
    raise ImportError(
        "OpenSlide is not installed. Install it with: "
        "pip install openslide-python openslide-bin"
    ) from exc


@dataclass(frozen=True)
class WSIInfo:
    """Basic metadata needed by the patching pipeline."""

    path: str
    width: int
    height: int
    level_count: int
    level_dimensions: Tuple[Tuple[int, int], ...]
    level_downsamples: Tuple[float, ...]
    properties: Dict[str, Any]


class WSIReader:
    """Small wrapper around OpenSlide for predictable WSI access."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"WSI not found: {self.path}")

        self.slide = openslide.open_slide(str(self.path))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.slide.close()

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self.slide.dimensions

    @property
    def level_count(self) -> int:
        return self.slide.level_count

    @property
    def level_dimensions(self):
        return tuple(self.slide.level_dimensions)

    @property
    def level_downsamples(self):
        return tuple(float(x) for x in self.slide.level_downsamples)

    @property
    def properties(self):
        return dict(self.slide.properties)

    def info(self) -> WSIInfo:
        return WSIInfo(
            path=str(self.path),
            width=int(self.dimensions[0]),
            height=int(self.dimensions[1]),
            level_count=int(self.level_count),
            level_dimensions=self.level_dimensions,
            level_downsamples=self.level_downsamples,
            properties=self.properties,
        )

    def thumbnail(
        self,
        max_width: int = 2048,
        max_height: int = 2048,
    ) -> Image.Image:
        """Create a bounded thumbnail instead of loading level 0."""
        return self.slide.get_thumbnail((max_width, max_height)).convert("RGB")

    def read_region(
        self,
        x: int,
        y: int,
        level: int,
        width: int,
        height: int,
    ) -> Image.Image:
        """Read only one rectangular region."""
        if level < 0 or level >= self.level_count:
            raise ValueError(
                f"Invalid level {level}; available levels: 0..{self.level_count - 1}"
            )

        if width <= 0 or height <= 0:
            raise ValueError("Region width and height must be positive.")

        return self.slide.read_region(
            (int(x), int(y)),
            int(level),
            (int(width), int(height)),
        ).convert("RGB")

    def read_region_array(
        self,
        x: int,
        y: int,
        level: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        return np.asarray(
            self.read_region(x, y, level, width, height)
        )


def is_supported_wsi(path: str | Path) -> bool:
    """Use OpenSlide's format detection without opening a full slide."""
    try:
        detected = openslide.OpenSlide.detect_format(str(path))
        return detected is not None
    except Exception:
        return False

from dataclasses import dataclass
from typing import Iterator, List

import numpy as np

from src.wsi.reader import WSIReader
from src.wsi.tissue import tissue_fraction


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    width: int
    height: int
    tissue_fraction: float


class WSITiler:
    """Generate level-0 tiles whose thumbnail region contains enough tissue."""

    def __init__(
        self,
        reader: WSIReader,
        tissue_mask: np.ndarray,
        tile_size: int = 256,
        stride: int | None = None,
        min_tissue_fraction: float = 0.25,
    ):
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")

        if stride is None:
            stride = tile_size

        if stride <= 0:
            raise ValueError("stride must be positive.")

        if not 0.0 <= min_tissue_fraction <= 1.0:
            raise ValueError("min_tissue_fraction must be between 0 and 1.")

        self.reader = reader
        self.mask = tissue_mask.astype(bool)
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.min_tissue_fraction = float(min_tissue_fraction)

        self.slide_width, self.slide_height = reader.dimensions

        self.mask_height, self.mask_width = self.mask.shape

        if self.mask_width == 0 or self.mask_height == 0:
            raise ValueError("Tissue mask cannot be empty.")

    def _level0_to_mask(self, x: int, y: int):
        mx = int(round(x * self.mask_width / self.slide_width))
        my = int(round(y * self.mask_height / self.slide_height))
        return mx, my

    def generate(self) -> Iterator[Tile]:
        """Yield tiles in deterministic row-major order."""
        for y in range(0, self.slide_height, self.stride):
            for x in range(0, self.slide_width, self.stride):
                # Read fixed-size patches only. Edge tiles are skipped to keep
                # every exported patch exactly tile_size x tile_size.
                if x + self.tile_size > self.slide_width:
                    continue
                if y + self.tile_size > self.slide_height:
                    continue

                mx0, my0 = self._level0_to_mask(x, y)
                mx1, my1 = self._level0_to_mask(
                    x + self.tile_size,
                    y + self.tile_size,
                )

                fraction = tissue_fraction(
                    self.mask,
                    mx0,
                    my0,
                    mx1,
                    my1,
                )

                if fraction >= self.min_tissue_fraction:
                    yield Tile(
                        x=x,
                        y=y,
                        width=self.tile_size,
                        height=self.tile_size,
                        tissue_fraction=fraction,
                    )

    def collect(self, max_tiles: int | None = None) -> List[Tile]:
        tiles = []

        for tile in self.generate():
            tiles.append(tile)

            if max_tiles is not None and len(tiles) >= max_tiles:
                break

        return tiles

    def read_tile(self, tile: Tile) -> np.ndarray:
        return self.reader.read_region_array(
            tile.x,
            tile.y,
            level=0,
            width=tile.width,
            height=tile.height,
        )

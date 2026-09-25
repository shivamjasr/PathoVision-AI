import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import DATA_DIR
from src.wsi.reader import WSIReader
from src.wsi.tiler import WSITiler
from src.wsi.tissue import make_tissue_mask, save_mask


def main():
    parser = argparse.ArgumentParser(
        description="Extract tissue-rich fixed-size patches from a WSI."
    )
    parser.add_argument("slide", type=str)
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "wsi_patches"),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--min-tissue",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=100,
        help="Safety limit for the first experiments.",
    )
    args = parser.parse_args()

    slide_path = Path(args.slide)

    if not slide_path.exists():
        raise FileNotFoundError(f"Slide not found: {slide_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = output_dir / "masks"
    patch_dir = output_dir / "patches"
    mask_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "tiles.csv"

    with WSIReader(slide_path) as reader:
        info = reader.info()

        print("WSI metadata")
        print(f"Path            : {info.path}")
        print(f"Dimensions      : {info.width} x {info.height}")
        print(f"Levels          : {info.level_count}")
        print(f"Level dimensions: {info.level_dimensions}")
        print(f"Downsamples     : {info.level_downsamples}")

        thumbnail = reader.thumbnail(
            max_width=args.thumbnail_size,
            max_height=args.thumbnail_size,
        )

        thumbnail_path = output_dir / "thumbnail.jpg"
        thumbnail.save(thumbnail_path, quality=92)

        mask = make_tissue_mask(thumbnail)
        mask_path = mask_dir / "tissue_mask.png"
        save_mask(mask, mask_path)

        tiler = WSITiler(
            reader=reader,
            tissue_mask=mask,
            tile_size=args.tile_size,
            stride=args.stride,
            min_tissue_fraction=args.min_tissue,
        )

        tiles = tiler.collect(max_tiles=args.max_tiles)

        print(f"\nSelected tiles: {len(tiles)}")

        with metadata_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "patch_id",
                    "x",
                    "y",
                    "width",
                    "height",
                    "tissue_fraction",
                    "path",
                ]
            )

            for index, tile in enumerate(tiles):
                patch_id = f"patch_{index:06d}"

                array = tiler.read_tile(tile)

                patch_path = patch_dir / f"{patch_id}.png"
                Image.fromarray(
                    np.asarray(array).astype(np.uint8),
                    mode="RGB",
                ).save(patch_path)

                writer.writerow(
                    [
                        patch_id,
                        tile.x,
                        tile.y,
                        tile.width,
                        tile.height,
                        f"{tile.tissue_fraction:.6f}",
                        str(patch_path),
                    ]
                )

        print(f"Thumbnail : {thumbnail_path}")
        print(f"Tissue mask: {mask_path}")
        print(f"Metadata  : {metadata_path}")
        print(f"Patches   : {patch_dir}")


if __name__ == "__main__":
    main()

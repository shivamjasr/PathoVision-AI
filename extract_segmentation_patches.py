import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import DATA_DIR
from src.wsi.reader import WSIReader
from src.wsi.tissue import make_tissue_mask, tissue_fraction
from src.segmentation.xml_annotations import (
    load_annotation_polygons,
    render_patch_mask,
    tumor_fraction,
)


def append_manifest_rows(manifest_path, rows):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "split",
        "image_path",
        "mask_path",
        "slide_id",
        "x",
        "y",
        "tumor_fraction",
    ]

    needs_header = (
        not manifest_path.exists()
        or manifest_path.stat().st_size == 0
    )

    with manifest_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if needs_header:
            writer.writeheader()

        writer.writerows(rows)


def generate_candidates(
    reader,
    tissue_mask,
    polygons,
    tile_size,
    stride,
    min_tissue,
):
    slide_width, slide_height = reader.dimensions
    mask_h, mask_w = tissue_mask.shape

    for y in range(
        0,
        slide_height - tile_size + 1,
        stride,
    ):
        for x in range(
            0,
            slide_width - tile_size + 1,
            stride,
        ):
            mx0 = int(
                round(x * mask_w / slide_width)
            )
            my0 = int(
                round(y * mask_h / slide_height)
            )
            mx1 = int(
                round((x + tile_size) * mask_w / slide_width)
            )
            my1 = int(
                round((y + tile_size) * mask_h / slide_height)
            )

            frac = tissue_fraction(
                tissue_mask,
                mx0,
                my0,
                mx1,
                my1,
            )

            if frac < min_tissue:
                continue

            # Level-0 WSI tile => level_downsample = 1.0
            tumor_mask = render_patch_mask(
                polygons,
                x=x,
                y=y,
                width=tile_size,
                height=tile_size,
                level_downsample=1.0,
            )

            yield x, y, frac, tumor_mask


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract segmentation image/mask patches from a WSI "
            "using contour annotations."
        )
    )
    parser.add_argument("--slide", required=True)
    parser.add_argument("--annotation-xml", required=True)
    parser.add_argument("--slide-id", required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        required=True,
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "segmentation"),
    )
    parser.add_argument(
        "--manifest",
        default=None,
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--min-tissue",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--min-tumor-fraction",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max-positive",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max-negative",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    if args.max_positive < 0 or args.max_negative < 0:
        raise ValueError("Maximum tile counts cannot be negative.")

    random.seed(args.seed)

    slide_path = Path(args.slide)
    annotation_path = Path(args.annotation_xml)

    if not slide_path.exists():
        raise FileNotFoundError(
            f"Slide not found: {slide_path}"
        )
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation XML not found: {annotation_path}"
        )

    output_dir = Path(args.output)
    images_dir = output_dir / "images" / args.split
    masks_dir = output_dir / "masks" / args.split

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    masks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_dir / "manifest.csv"
    )

    polygons = load_annotation_polygons(
        annotation_path
    )

    positive_candidates = []
    negative_candidates = []

    with WSIReader(slide_path) as reader:
        thumbnail = reader.thumbnail(
            max_width=2048,
            max_height=2048,
        )

        tissue_mask = make_tissue_mask(
            thumbnail
        )

        for item in generate_candidates(
            reader,
            tissue_mask,
            polygons,
            args.tile_size,
            args.stride,
            args.min_tissue,
        ):
            x, y, tissue_frac, tumor_mask = item
            frac = tumor_fraction(tumor_mask)

            if frac >= args.min_tumor_fraction:
                positive_candidates.append(item)
            else:
                negative_candidates.append(item)

        random.shuffle(positive_candidates)
        random.shuffle(negative_candidates)

        positive_selected = positive_candidates[
            : args.max_positive
        ]
        negative_selected = negative_candidates[
            : args.max_negative
        ]

        selected = (
            [("positive", item) for item in positive_selected]
            + [("negative", item) for item in negative_selected]
        )
        random.shuffle(selected)

        rows = []

        for index, (_, item) in enumerate(selected):
            x, y, tissue_frac, tumor_mask = item

            patch = reader.read_region_array(
                x=x,
                y=y,
                level=0,
                width=args.tile_size,
                height=args.tile_size,
            )

            patch_id = (
                f"{args.slide_id}_{args.split}_{index:06d}"
            )

            image_path = (
                images_dir / f"{patch_id}.png"
            )
            mask_path = (
                masks_dir / f"{patch_id}.png"
            )

            Image.fromarray(
                patch.astype(np.uint8),
                mode="RGB",
            ).save(image_path)

            Image.fromarray(
                tumor_mask.astype(np.uint8),
                mode="L",
            ).save(mask_path)

            rows.append(
                {
                    "split": args.split,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "slide_id": args.slide_id,
                    "x": x,
                    "y": y,
                    "tumor_fraction": f"{tumor_fraction(tumor_mask):.6f}",
                }
            )

    append_manifest_rows(
        manifest_path,
        rows,
    )

    print("\nSegmentation patch extraction complete.")
    print(f"Positive candidates: {len(positive_candidates):,}")
    print(f"Negative candidates: {len(negative_candidates):,}")
    print(f"Selected patches   : {len(rows):,}")
    print(f"Manifest            : {manifest_path}")
    print(f"Images              : {images_dir}")
    print(f"Masks               : {masks_dir}")


if __name__ == "__main__":
    main()

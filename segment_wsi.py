import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.config import DATA_DIR, get_device, make_output_dirs
from src.segmentation.quantification import (
    clean_binary_mask,
    quantify_tumor,
    save_mask,
    stitch_masks_to_slide,
)
from src.segmentation.wsi_inference import (
    load_unet,
    predict_patch_masks,
    read_manifest,
)
from src.wsi.reader import WSIReader


def save_overlay(
    thumbnail_path,
    mask,
    output_path,
    alpha=0.45,
):
    thumbnail = Image.open(
        thumbnail_path
    ).convert("RGB")

    if thumbnail.size != (
        mask.shape[1],
        mask.shape[0],
    ):
        raise ValueError(
            "Thumbnail and stitched mask sizes do not match."
        )

    fig, ax = plt.subplots(
        figsize=(
            max(4, thumbnail.width / 220),
            max(4, thumbnail.height / 220),
        )
    )

    ax.imshow(thumbnail)

    ax.imshow(
        mask > 0,
        alpha=alpha,
    )

    ax.set_title(
        "WSI U-Net tumor segmentation"
    )
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_regions_csv(regions, path):
    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "region_id",
                "area_pixels_thumbnail",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
                "centroid_x",
                "centroid_y",
            ]
        )

        for region in regions:
            writer.writerow(
                [
                    region.region_id,
                    region.area_pixels,
                    region.bbox_x,
                    region.bbox_y,
                    region.bbox_width,
                    region.bbox_height,
                    f"{region.centroid_x:.3f}",
                    f"{region.centroid_y:.3f}",
                ]
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run U-Net over WSI patches, stitch the masks, "
            "clean the result, and quantify tumor regions."
        )
    )

    parser.add_argument(
        "--slide",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Segmentation manifest containing image_path, x, y, etc.",
    )
    parser.add_argument(
        "--thumbnail",
        required=True,
        help="Thumbnail matching the analyzed slide.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument(
        "--split",
        default="val",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--open-kernel",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--min-region-area",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--output",
        default=str(
            DATA_DIR / "wsi_segmentation"
        ),
    )

    args = parser.parse_args()

    make_output_dirs()

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    thumbnail_path = Path(
        args.thumbnail
    )

    if not thumbnail_path.exists():
        raise FileNotFoundError(
            f"Thumbnail not found: {thumbnail_path}"
        )

    records = read_manifest(
        args.manifest,
        split=args.split,
    )

    device = get_device()
    print(f"Device: {device}")
    print(
        f"Patches for inference: {len(records):,}"
    )

    model, checkpoint = load_unet(
        args.checkpoint,
        device=device,
    )

    checkpoint_image_size = checkpoint.get(
        "image_size",
        args.image_size,
    )

    patch_masks = predict_patch_masks(
        records=records,
        model=model,
        device=device,
        image_size=checkpoint_image_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
        num_workers=args.num_workers,
    )

    with WSIReader(args.slide) as reader:
        slide_width, slide_height = (
            reader.dimensions
        )

    thumbnail = Image.open(
        thumbnail_path
    ).convert("RGB")

    stitched_fraction, coverage = (
        stitch_masks_to_slide(
            records=records,
            patch_masks=patch_masks,
            slide_width=slide_width,
            slide_height=slide_height,
            thumbnail_size=thumbnail.size,
        )
    )

    # Fraction > 0.5 means that for overlapping predictions at least half
    # of the contributing patch pixels voted positive.
    raw_binary = (
        stitched_fraction
        >= 0.5
    ).astype(np.uint8) * 255

    cleaned = clean_binary_mask(
        raw_binary,
        open_kernel=args.open_kernel,
        close_kernel=args.close_kernel,
        min_component_area=args.min_region_area,
    )

    # Use the Phase-2 tissue mask when present and aligned with thumbnail.
    tissue_mask = None

    candidate_mask = (
        thumbnail_path.parent
        / "masks"
        / "tissue_mask.png"
    )

    if candidate_mask.exists():
        tissue_mask_image = Image.open(
            candidate_mask
        ).convert("L")

        if tissue_mask_image.size == thumbnail.size:
            tissue_mask = (
                np.asarray(
                    tissue_mask_image
                )
                > 0
            )

    quantification = quantify_tumor(
        cleaned_mask=cleaned,
        coverage_map=coverage,
        tissue_mask=tissue_mask,
        region_min_area=args.min_region_area,
    )

    save_mask(
        raw_binary,
        output / "raw_stitched_mask.png",
    )
    save_mask(
        cleaned,
        output / "cleaned_tumor_mask.png",
    )

    np.save(
        output / "stitched_mask_fraction.npy",
        stitched_fraction,
    )

    np.save(
        output / "coverage_map.npy",
        coverage,
    )

    overlay_path = (
        output / "tumor_segmentation_overlay.png"
    )

    save_overlay(
        thumbnail_path,
        cleaned,
        overlay_path,
    )

    regions = quantification.pop(
        "regions"
    )

    save_regions_csv(
        regions,
        output / "tumor_regions.csv",
    )

    quantification.update(
        {
            "slide": str(
                Path(args.slide).resolve()
            ),
            "manifest": str(
                Path(args.manifest).resolve()
            ),
            "checkpoint": str(
                Path(
                    args.checkpoint
                ).resolve()
            )
            if args.checkpoint
            else str(
                (
                    Path(
                        "checkpoints"
                    )
                    / "best_unet.pt"
                )
                .resolve()
            ),
            "split": args.split,
            "device": str(device),
            "patches_inferred": len(records),
            "threshold": args.threshold,
            "open_kernel": args.open_kernel,
            "close_kernel": args.close_kernel,
            "min_region_area": args.min_region_area,
            "slide_width": int(slide_width),
            "slide_height": int(slide_height),
            "thumbnail_width": int(
                thumbnail.width
            ),
            "thumbnail_height": int(
                thumbnail.height
            ),
            "region_areas_pixels_thumbnail": [
                region.area_pixels
                for region in regions
            ],
        }
    )

    summary_path = (
        output
        / "tumor_quantification.json"
    )

    summary_path.write_text(
        json.dumps(
            quantification,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nWSI segmentation complete.")
    print(
        f"Tumor area (thumbnail px): "
        f"{quantification['tumor_area_pixels_thumbnail']:,}"
    )
    print(
        f"Analyzable area (thumbnail px): "
        f"{quantification['analyzable_area_pixels_thumbnail']:,}"
    )
    print(
        f"Tumor percentage of analyzable area: "
        f"{quantification['tumor_percentage_of_analyzable_area']:.2f}%"
    )
    print(
        f"Detected tumor regions: "
        f"{quantification['region_count']}"
    )
    print(
        f"Largest region (thumbnail px): "
        f"{quantification['largest_region_pixels_thumbnail']:,}"
    )
    print(
        f"Cleaned mask: {output / 'cleaned_tumor_mask.png'}"
    )
    print(
        f"Overlay: {overlay_path}"
    )
    print(
        f"Regions CSV: {output / 'tumor_regions.csv'}"
    )
    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()

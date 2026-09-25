from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.config import get_device
from src.segmentation.quantification import (
    clean_binary_mask,
    quantify_tumor,
    save_mask,
    stitch_masks_to_slide,
)
from src.segmentation.wsi_inference import (
    load_unet,
    predict_patch_masks,
)
from src.wsi.reader import WSIReader


def run_segmentation_from_tiles(
    *,
    metadata_path: Path,
    thumbnail_path: Path,
    slide_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    tissue_mask: np.ndarray | None,
    image_size: int = 256,
    batch_size: int = 4,
    threshold: float = 0.5,
    min_region_area: int = 30,
) -> dict:
    records: list[dict[str, str]] = []

    with metadata_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            records.append(
                {
                    "image_path": row["path"],
                    "x": row["x"],
                    "y": row["y"],
                    "width": row["width"],
                    "height": row["height"],
                }
            )

    if not records:
        raise ValueError("No extracted tile records available for segmentation.")

    device = get_device()
    model, checkpoint = load_unet(checkpoint_path, device=device)
    checkpoint_image_size = int(checkpoint.get("image_size", image_size))

    patch_masks = predict_patch_masks(
        records=records,
        model=model,
        device=device,
        image_size=checkpoint_image_size,
        batch_size=batch_size,
        threshold=threshold,
        num_workers=0,
    )

    thumbnail = Image.open(thumbnail_path).convert("RGB")
    with WSIReader(slide_path) as reader:
        slide_width, slide_height = reader.dimensions

    stitched_fraction, coverage = stitch_masks_to_slide(
        records=records,
        patch_masks=patch_masks,
        slide_width=slide_width,
        slide_height=slide_height,
        thumbnail_size=thumbnail.size,
    )

    raw_binary = (stitched_fraction >= 0.5).astype(np.uint8) * 255
    cleaned = clean_binary_mask(
        raw_binary,
        open_kernel=3,
        close_kernel=7,
        min_component_area=min_region_area,
    )

    quantification = quantify_tumor(
        cleaned_mask=cleaned,
        coverage_map=coverage,
        tissue_mask=tissue_mask,
        region_min_area=min_region_area,
    )
    regions = quantification.pop("regions")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_mask(raw_binary, output_dir / "raw_stitched_mask.png")
    save_mask(cleaned, output_dir / "cleaned_tumor_mask.png")
    np.save(output_dir / "stitched_mask_fraction.npy", stitched_fraction)
    np.save(output_dir / "segmentation_coverage_map.npy", coverage)

    overlay_path = output_dir / "tumor_segmentation_overlay.png"
    fig, ax = plt.subplots(
        figsize=(max(4, thumbnail.width / 220), max(4, thumbnail.height / 220))
    )
    ax.imshow(thumbnail)
    ax.imshow(cleaned > 0, alpha=0.45)
    ax.set_title("WSI U-Net tumor segmentation")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    regions_path = output_dir / "tumor_regions.csv"
    with regions_path.open("w", newline="", encoding="utf-8") as file:
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

    summary = {
        **quantification,
        "device": str(device),
        "patches_inferred": len(records),
        "threshold": threshold,
        "min_region_area": min_region_area,
        "slide_width": int(slide_width),
        "slide_height": int(slide_height),
        "thumbnail_width": int(thumbnail.width),
        "thumbnail_height": int(thumbnail.height),
        "region_areas_pixels_thumbnail": [r.area_pixels for r in regions],
        "artifacts": {
            "raw_mask": str(output_dir / "raw_stitched_mask.png"),
            "cleaned_mask": str(output_dir / "cleaned_tumor_mask.png"),
            "overlay": str(overlay_path),
            "regions": str(regions_path),
            "quantification": str(output_dir / "tumor_quantification.json"),
        },
        "clinical_note": (
            "Segmentation is a research/engineering output and is not a clinically validated diagnosis."
        ),
    }

    summary_path = output_dir / "tumor_quantification.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

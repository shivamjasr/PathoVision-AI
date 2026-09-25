from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Region:
    region_id: int
    area_pixels: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    centroid_x: float
    centroid_y: float


def stitch_masks_to_slide(
    records,
    patch_masks,
    slide_width,
    slide_height,
    thumbnail_size,
):
    """Stitch patch masks into a thumbnail-resolution probability/coverage map.

    Each patch mask is resized using nearest-neighbor interpolation and
    projected into thumbnail coordinates. Overlaps are accumulated and
    averaged.
    """
    if len(records) != len(patch_masks):
        raise ValueError(
            "records and patch_masks must have identical lengths."
        )

    thumb_width, thumb_height = thumbnail_size

    mask_sum = np.zeros(
        (thumb_height, thumb_width),
        dtype=np.float32,
    )

    coverage = np.zeros(
        (thumb_height, thumb_width),
        dtype=np.float32,
    )

    for row, mask in zip(
        records,
        patch_masks,
    ):
        x = int(row["x"])
        y = int(row["y"])
        width = int(row["width"])
        height = int(row["height"])

        target_x0 = int(
            round(x * thumb_width / slide_width)
        )
        target_y0 = int(
            round(y * thumb_height / slide_height)
        )

        target_x1 = int(
            round(
                (x + width)
                * thumb_width
                / slide_width
            )
        )
        target_y1 = int(
            round(
                (y + height)
                * thumb_height
                / slide_height
            )
        )

        target_x0 = max(
            0,
            min(target_x0, thumb_width),
        )
        target_x1 = max(
            0,
            min(target_x1, thumb_width),
        )
        target_y0 = max(
            0,
            min(target_y0, thumb_height),
        )
        target_y1 = max(
            0,
            min(target_y1, thumb_height),
        )

        if (
            target_x1 <= target_x0
            or target_y1 <= target_y0
        ):
            continue

        target_width = (
            target_x1 - target_x0
        )
        target_height = (
            target_y1 - target_y0
        )

        resized = cv2.resize(
            mask,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

        binary = (
            resized > 0
        ).astype(np.float32)

        mask_sum[
            target_y0:target_y1,
            target_x0:target_x1,
        ] += binary

        coverage[
            target_y0:target_y1,
            target_x0:target_x1,
        ] += 1.0

    stitched_fraction = np.zeros_like(
        mask_sum
    )

    covered = coverage > 0
    stitched_fraction[covered] = (
        mask_sum[covered]
        / coverage[covered]
    )

    return stitched_fraction, coverage


def clean_binary_mask(
    mask,
    open_kernel=3,
    close_kernel=7,
    min_component_area=30,
):
    """Apply small morphological cleanup operations."""
    binary = (
        mask > 0.5
    ).astype(np.uint8) * 255

    if open_kernel >= 3:
        if open_kernel % 2 == 0:
            open_kernel += 1

        kernel = np.ones(
            (open_kernel, open_kernel),
            dtype=np.uint8,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )

    if close_kernel >= 3:
        if close_kernel % 2 == 0:
            close_kernel += 1

        kernel = np.ones(
            (close_kernel, close_kernel),
            dtype=np.uint8,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
        )

    if min_component_area > 0:
        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        filtered = np.zeros_like(
            binary
        )

        for label_id in range(
            1,
            num_labels,
        ):
            area = int(
                stats[
                    label_id,
                    cv2.CC_STAT_AREA,
                ]
            )

            if area >= min_component_area:
                filtered[
                    labels == label_id
                ] = 255

        binary = filtered

    return binary


def extract_regions(
    binary_mask,
    min_area=30,
):
    """Extract connected tumor regions."""
    binary = (
        binary_mask > 0
    ).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    regions = []

    next_region_id = 1

    for label_id in range(
        1,
        num_labels,
    ):
        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            continue

        bbox_x = int(
            stats[
                label_id,
                cv2.CC_STAT_LEFT,
            ]
        )
        bbox_y = int(
            stats[
                label_id,
                cv2.CC_STAT_TOP,
            ]
        )
        bbox_width = int(
            stats[
                label_id,
                cv2.CC_STAT_WIDTH,
            ]
        )
        bbox_height = int(
            stats[
                label_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        centroid_x = float(
            centroids[label_id][0]
        )
        centroid_y = float(
            centroids[label_id][1]
        )

        regions.append(
            Region(
                region_id=next_region_id,
                area_pixels=area,
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
            )
        )

        next_region_id += 1

    return regions


def quantify_tumor(
    cleaned_mask,
    coverage_map,
    tissue_mask=None,
    region_min_area=30,
):
    """Calculate slide-level area statistics.

    All area values refer to the thumbnail grid when a thumbnail is supplied.
    """
    tumor = cleaned_mask > 0
    regions = extract_regions(
        cleaned_mask,
        min_area=region_min_area,
    )

    covered = coverage_map > 0

    if tissue_mask is not None:
        tissue = (
            tissue_mask.astype(bool)
        )
        analyzable = covered & tissue
    else:
        analyzable = covered

    tumor_area = int(
        (tumor & analyzable).sum()
    )

    analyzable_area = int(
        analyzable.sum()
    )

    covered_area = int(
        covered.sum()
    )

    if analyzable_area > 0:
        tumor_fraction = (
            tumor_area
            / analyzable_area
        )
    else:
        tumor_fraction = 0.0

    return {
        "tumor_area_pixels_thumbnail": tumor_area,
        "analyzable_area_pixels_thumbnail": analyzable_area,
        "covered_area_pixels_thumbnail": covered_area,
        "tumor_fraction_of_analyzable_area": float(
            tumor_fraction
        ),
        "tumor_percentage_of_analyzable_area": float(
            100.0 * tumor_fraction
        ),
        "region_count": len(regions),
        "largest_region_pixels_thumbnail": (
            max(
                (region.area_pixels for region in regions),
                default=0,
            )
        ),
        "regions": regions,
    }


def save_mask(mask, path):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        mask.astype(np.uint8),
        mode="L",
    ).save(path)

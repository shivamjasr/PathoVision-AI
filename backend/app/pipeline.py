from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np

from backend.app.config import (
    CLASSIFIER_CHECKPOINT,
    DEFAULT_CLASSIFIER_BATCH,
    DEFAULT_MIN_TISSUE,
    DEFAULT_SEGMENTATION_BATCH,
    DEFAULT_TILE_SIZE,
    UNET_CHECKPOINT,
)


def _project_path(value: str | Path):
    return Path(value)


def run_analysis(
    *,
    job,
    job_store,
    slide_path: Path,
    mode: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    min_tissue: float = DEFAULT_MIN_TISSUE,
):
    """Run the local PathoVision pipeline.

    `mode=classify`:
        WSI → tissue tiles → ResNet18 → heatmap.

    `mode=full`:
        Runs classification/heatmap and then reuses the same extracted WSI
        tiles for U-Net segmentation + slide-level quantification when the
        segmentation checkpoint is available.
    """
    job_store.update(
        job,
        status="running",
        progress=3,
        stage="validate",
        message="Validating the uploaded slide.",
    )

    if not slide_path.exists():
        raise FileNotFoundError(
            f"Slide not found: {slide_path}"
        )

    mode = mode.lower()

    if mode not in {"classify", "full"}:
        raise ValueError("Supported analysis modes are 'classify' and 'full'.")

    # Reuse the project's WSI extraction implementation.
    from src.wsi.reader import WSIReader
    from src.wsi.tissue import make_tissue_mask
    from src.wsi.tiler import WSITiler

    work_dir = (
        slide_path.parent.parent
        / "analysis"
        / job
    )
    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_dir = work_dir / "patches"
    patch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    job_store.update(
        job,
        progress=8,
        stage="read_slide",
        message="Reading WSI metadata and creating a low-resolution thumbnail.",
    )

    with WSIReader(slide_path) as reader:
        info = reader.info()
        thumbnail = reader.thumbnail(
            max_width=2048,
            max_height=2048,
        )

        thumbnail_path = (
            work_dir / "thumbnail.jpg"
        )
        thumbnail.save(
            thumbnail_path,
            quality=92,
        )

        job_store.update(
            job,
            progress=15,
            stage="tissue_detection",
            message="Detecting tissue-rich regions.",
        )

        tissue_mask = make_tissue_mask(
            thumbnail
        )

        tiler = WSITiler(
            reader=reader,
            tissue_mask=tissue_mask,
            tile_size=tile_size,
            stride=tile_size,
            min_tissue_fraction=min_tissue,
        )

        tiles = tiler.collect(
            max_tiles=500
        )

        if not tiles:
            raise RuntimeError(
                "No tissue-rich tiles were found. "
                "Try a different slide or lower --min-tissue."
            )

        job_store.update(
            job,
            progress=22,
            stage="extract_tiles",
            message=f"Extracting {len(tiles)} tissue-rich tiles.",
        )

        metadata_path = (
            work_dir / "tiles.csv"
        )

        with metadata_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
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

            for index, tile in enumerate(
                tiles
            ):
                patch_id = (
                    f"patch_{index:06d}"
                )

                patch = tiler.read_tile(
                    tile
                )

                patch_path = (
                    patch_dir
                    / f"{patch_id}.png"
                )

                from PIL import Image

                Image.fromarray(
                    np.asarray(patch).astype(
                        np.uint8
                    ),
                    mode="RGB",
                ).save(
                    patch_path
                )

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

    job_store.update(
        job,
        progress=30,
        stage="classifier_inference",
        message="Running ResNet18 inference over WSI tiles.",
    )

    if not CLASSIFIER_CHECKPOINT.exists():
        raise FileNotFoundError(
            "ResNet18 checkpoint is missing: "
            f"{CLASSIFIER_CHECKPOINT}"
        )

    from src.inference.heatmap import (
        build_probability_maps,
        save_heatmap,
        save_overlay,
    )
    from src.inference.wsi_inference import (
        load_resnet_checkpoint,
        predict_tiles,
        read_tile_records,
    )

    records = read_tile_records(
        metadata_path
    )

    device = None

    from src.config import get_device

    device = get_device()

    model, checkpoint = load_resnet_checkpoint(
        CLASSIFIER_CHECKPOINT,
        device=device,
    )

    job_store.update(
        job,
        progress=38,
        stage="classifier_inference",
        message=f"Scoring {len(records)} WSI tiles.",
    )

    probabilities = predict_tiles(
        records,
        model,
        device,
        batch_size=DEFAULT_CLASSIFIER_BATCH,
        num_workers=0,
    )

    predictions_csv = (
        work_dir
        / "tile_predictions.csv"
    )

    with predictions_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "patch_id",
                "x",
                "y",
                "width",
                "height",
                "tissue_fraction",
                "tumor_probability",
            ]
        )

        for record, probability in zip(
            records,
            probabilities,
        ):
            writer.writerow(
                [
                    record.patch_id,
                    record.x,
                    record.y,
                    record.width,
                    record.height,
                    f"{record.tissue_fraction:.6f}",
                    f"{float(probability):.6f}",
                ]
            )

    job_store.update(
        job,
        progress=60,
        stage="build_heatmap",
        message="Reconstructing the WSI probability heatmap.",
    )

    xs = [r.x for r in records]
    ys = [r.y for r in records]
    widths = [r.width for r in records]
    heights = [r.height for r in records]

    probability_map, coverage = (
        build_probability_maps(
            x=xs,
            y=ys,
            width=widths,
            height=heights,
            probabilities=probabilities,
            slide_width=info.width,
            slide_height=info.height,
            thumbnail_size=thumbnail.size,
        )
    )

    np.save(
        work_dir / "probability_map.npy",
        probability_map,
    )
    np.save(
        work_dir / "coverage_map.npy",
        coverage,
    )

    heatmap_path = (
        work_dir
        / "tumor_probability_heatmap.png"
    )
    overlay_path = (
        work_dir
        / "tumor_heatmap_overlay.png"
    )

    save_heatmap(
        probability_map,
        heatmap_path,
    )

    save_overlay(
        thumbnail_path,
        probability_map,
        overlay_path,
        alpha=0.45,
    )

    threshold = 0.5
    positive = probabilities >= threshold

    segmentation_summary = None
    segmentation_dir = work_dir / "segmentation"

    if mode == "full":
        job_store.update(
            job,
            progress=66,
            stage="segmentation_inference",
            message="Running U-Net segmentation on the extracted WSI tiles.",
        )

        if not UNET_CHECKPOINT.exists():
            job_store.update(
                job,
                progress=66,
                stage="segmentation_skipped",
                message=(
                    "U-Net checkpoint is missing; classification completed "
                    "and segmentation was skipped."
                ),
            )
        else:
            from backend.app.segmentation_pipeline import run_segmentation_from_tiles

            segmentation_summary = run_segmentation_from_tiles(
                metadata_path=metadata_path,
                thumbnail_path=thumbnail_path,
                slide_path=slide_path,
                checkpoint_path=UNET_CHECKPOINT,
                output_dir=segmentation_dir,
                tissue_mask=tissue_mask,
                image_size=256,
                batch_size=DEFAULT_SEGMENTATION_BATCH,
                threshold=0.5,
                min_region_area=30,
            )
            job_store.update(
                job,
                progress=88,
                stage="quantification",
                message="Quantifying segmented tumor regions.",
            )

    summary = {
        "job_id": job,
        "mode": mode,
        "slide": info.path,
        "slide_dimensions": [
            info.width,
            info.height,
        ],
        "pyramid_levels": info.level_count,
        "level_dimensions": [
            list(item)
            for item in info.level_dimensions
        ],
        "level_downsamples": list(
            info.level_downsamples
        ),
        "tiles_inferred": int(
            len(records)
        ),
        "positive_tiles": int(
            positive.sum()
        ),
        "positive_tile_fraction": float(
            positive.mean()
        ),
        "mean_tumor_probability": float(
            probabilities.mean()
        ),
        "max_tumor_probability": float(
            probabilities.max()
        ),
        "threshold": threshold,
        "thumbnail_size": [
            thumbnail.width,
            thumbnail.height,
        ],
        "artifacts": {
            "thumbnail": str(
                thumbnail_path
            ),
            "tile_predictions": str(
                predictions_csv
            ),
            "heatmap": str(
                heatmap_path
            ),
            "overlay": str(
                overlay_path
            ),
            "probability_map": str(
                work_dir
                / "probability_map.npy"
            ),
            "coverage_map": str(
                work_dir
                / "coverage_map.npy"
            ),
        },
        "segmentation": segmentation_summary,
        "clinical_note": (
            "This output is a machine-learning research/engineering "
            "artifact, not a clinically validated diagnosis."
        ),
    }

    summary_path = work_dir / "summary.json"

    # Keep a copy of the visual artifacts under the API runtime root
    # so the frontend can access them through a controlled endpoint.
    public_dir = work_dir / "public"
    public_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts_to_publish = [
        thumbnail_path,
        heatmap_path,
        overlay_path,
        predictions_csv,
        summary_path,
    ]

    if segmentation_summary is not None:
        for name in (
            "raw_stitched_mask.png",
            "cleaned_tumor_mask.png",
            "tumor_segmentation_overlay.png",
            "tumor_regions.csv",
            "tumor_quantification.json",
        ):
            candidate = segmentation_dir / name
            if candidate.exists():
                artifacts_to_publish.append(candidate)

    for artifact in artifacts_to_publish:
        shutil.copy2(artifact, public_dir / artifact.name)

    # The application exposes only public, browser-safe artifacts through the API.
    summary["published_artifacts"] = [artifact.name for artifact in artifacts_to_publish]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.copy2(summary_path, public_dir / summary_path.name)

    job_store.update(
        job,
        status="completed",
        progress=100,
        stage="completed",
        message="Analysis completed.",
        result=summary,
    )

    return summary

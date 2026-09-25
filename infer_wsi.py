import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.config import DATA_DIR, get_device, make_output_dirs
from src.inference.heatmap import build_probability_maps, save_heatmap, save_overlay
from src.inference.wsi_inference import load_resnet_checkpoint, predict_tiles, read_tile_records
from src.wsi.reader import WSIReader


def main():
    parser = argparse.ArgumentParser(description="Run ResNet18 over WSI tiles and create a tumor-probability heatmap.")
    parser.add_argument("--slide", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--thumbnail", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=str(DATA_DIR / "wsi_inference"))
    args = parser.parse_args()

    make_output_dirs()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    records = read_tile_records(args.metadata)
    model, checkpoint, checkpoint_path = load_resnet_checkpoint(args.checkpoint, device)
    probabilities = predict_tiles(records, model, device, args.batch_size, args.num_workers)

    thumbnail_path = Path(args.thumbnail)
    if not thumbnail_path.exists():
        raise FileNotFoundError(f"Thumbnail not found: {thumbnail_path}")

    from PIL import Image
    thumbnail = Image.open(thumbnail_path).convert("RGB")

    with WSIReader(args.slide) as reader:
        slide_width, slide_height = reader.dimensions

    probability_map, coverage = build_probability_maps(
        [r.x for r in records], [r.y for r in records],
        [r.width for r in records], [r.height for r in records], probabilities,
        slide_width, slide_height, thumbnail.size
    )

    predictions_csv = output / "tile_predictions.csv"
    with predictions_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patch_id","x","y","width","height","tissue_fraction","tumor_probability","predicted_tumor"])
        for record, prob in zip(records, probabilities):
            writer.writerow([record.patch_id, record.x, record.y, record.width, record.height, f"{record.tissue_fraction:.6f}", f"{float(prob):.6f}", int(prob >= args.threshold)])

    np.save(output / "tumor_probability_map.npy", probability_map)
    np.save(output / "coverage_map.npy", coverage)

    heatmap_path = save_heatmap(probability_map, output / "tumor_probability_heatmap.png")
    overlay_path = save_overlay(thumbnail_path, probability_map, output / "tumor_heatmap_overlay.png")

    top_k = min(20, len(records))
    order = np.argsort(-probabilities)[:top_k]
    with (output / "top_positive_tiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["rank","patch_id","x","y","tumor_probability","patch_path"])
        for rank, idx in enumerate(order, start=1):
            r = records[int(idx)]
            writer.writerow([rank, r.patch_id, r.x, r.y, f"{float(probabilities[idx]):.6f}", str(r.path)])

    positive = probabilities >= args.threshold
    summary = {
        "slide": str(Path(args.slide).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "tiles_inferred": int(len(records)),
        "positive_tiles": int(positive.sum()),
        "positive_tile_fraction": float(positive.mean()),
        "threshold": float(args.threshold),
        "mean_probability": float(probabilities.mean()),
        "max_probability": float(probabilities.max()),
        "slide_width": int(slide_width),
        "slide_height": int(slide_height),
        "thumbnail_width": int(thumbnail.width),
        "thumbnail_height": int(thumbnail.height),
        "model_epoch": checkpoint.get("epoch"),
        "model_strategy": checkpoint.get("strategy"),
    }
    (output / "wsi_inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nWSI inference complete.")
    print(f"Tiles inferred : {len(records):,}")
    print(f"Positive tiles : {int(positive.sum()):,}")
    print(f"Mean probability: {probabilities.mean():.4f}")
    print(f"Max probability : {probabilities.max():.4f}")
    print(f"Predictions     : {predictions_csv}")
    print(f"Heatmap         : {heatmap_path}")
    print(f"Overlay         : {overlay_path}")


if __name__ == "__main__":
    main()

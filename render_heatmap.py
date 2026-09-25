import argparse
import csv
from pathlib import Path

from src.inference.heatmap import build_probability_maps, save_heatmap, save_overlay
from src.wsi.reader import WSIReader


def main():
    parser = argparse.ArgumentParser(description="Regenerate WSI heatmap images from tile_predictions.csv.")
    parser.add_argument("--slide", required=True)
    parser.add_argument("--thumbnail", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="results/plots/heatmap_render")
    args = parser.parse_args()

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    thumbnail = Image.open(args.thumbnail).convert("RGB")

    with Path(args.predictions).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Prediction CSV is empty.")

    with WSIReader(args.slide) as reader:
        slide_width, slide_height = reader.dimensions

    probability_map, _ = build_probability_maps(
        [int(r["x"]) for r in rows], [int(r["y"]) for r in rows],
        [int(r["width"]) for r in rows], [int(r["height"]) for r in rows],
        [float(r["tumor_probability"]) for r in rows],
        slide_width, slide_height, thumbnail.size
    )

    save_heatmap(probability_map, output / "tumor_probability_heatmap.png")
    save_overlay(args.thumbnail, probability_map, output / "tumor_heatmap_overlay.png")
    print(f"Saved visualizations to: {output}")


if __name__ == "__main__":
    main()

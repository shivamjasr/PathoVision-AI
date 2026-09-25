import argparse
import json
from pathlib import Path

from src.wsi.reader import WSIReader, is_supported_wsi


def main():
    parser = argparse.ArgumentParser(
        description="Inspect WSI dimensions, pyramid levels and properties."
    )
    parser.add_argument("slide", type=str)
    args = parser.parse_args()

    slide_path = Path(args.slide)

    if not slide_path.exists():
        raise FileNotFoundError(f"Slide not found: {slide_path}")

    print(f"OpenSlide format detected: {is_supported_wsi(slide_path)}")

    with WSIReader(slide_path) as reader:
        info = reader.info()

    print("\n===== WSI INFO =====")
    print(f"File: {info.path}")
    print(f"Base dimensions: {info.width:,} x {info.height:,}")
    print(f"Level count: {info.level_count}")

    for idx, (dims, downsample) in enumerate(
        zip(info.level_dimensions, info.level_downsamples)
    ):
        print(
            f"Level {idx}: {dims[0]:,} x {dims[1]:,} "
            f"| downsample={downsample}"
        )

    print("\nSelected properties:")
    interesting = (
        "openslide.vendor",
        "openslide.objective-power",
        "openslide.mpp-x",
        "openslide.mpp-y",
    )

    selected = {}
    for key in interesting:
        if key in info.properties:
            selected[key] = info.properties[key]
            print(f"{key}: {info.properties[key]}")

    output = Path("results/metrics/wsi_info.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "path": info.path,
        "width": info.width,
        "height": info.height,
        "level_count": info.level_count,
        "level_dimensions": [list(x) for x in info.level_dimensions],
        "level_downsamples": list(info.level_downsamples),
        "selected_properties": selected,
    }

    output.write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved metadata: {output}")


if __name__ == "__main__":
    main()

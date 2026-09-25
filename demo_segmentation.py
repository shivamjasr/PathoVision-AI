from pathlib import Path

import numpy as np
from PIL import Image

from src.segmentation.quantification import (
    clean_binary_mask,
    quantify_tumor,
    stitch_masks_to_slide,
)


def main():
    output = Path("results/phase5_demo")
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # A 512x512 synthetic slide with four 128x128 patches.
    slide_width = 512
    slide_height = 512

    # Synthetic thumbnail tissue mask.
    tissue = np.zeros(
        (256, 256),
        dtype=bool,
    )

    yy, xx = np.mgrid[
        0:256,
        0:256,
    ]

    tissue[
        ((xx - 128) ** 2)
        / (115 ** 2)
        + ((yy - 128) ** 2)
        / (100 ** 2)
        <= 1
    ] = True

    # Define four patch records.
    records = []

    for idx, (x, y) in enumerate(
        [
            (0, 0),
            (128, 0),
            (256, 128),
            (128, 256),
        ]
    ):
        records.append(
            {
                "patch_id": f"demo_{idx}",
                "x": str(x),
                "y": str(y),
                "width": "128",
                "height": "128",
            }
        )

    masks = []

    for idx in range(4):
        mask = np.zeros(
            (128, 128),
            dtype=np.uint8,
        )

        if idx in (1, 2):
            yy2, xx2 = np.mgrid[
                0:128,
                0:128,
            ]

            mask[
                (
                    (xx2 - 64) ** 2
                    + (yy2 - 64) ** 2
                    <= 38 ** 2
                )
            ] = 255

        masks.append(mask)

    fraction, coverage = (
        stitch_masks_to_slide(
            records,
            masks,
            slide_width,
            slide_height,
            (256, 256),
        )
    )

    raw = (
        fraction >= 0.5
    ).astype(np.uint8) * 255

    cleaned = clean_binary_mask(
        raw,
        open_kernel=3,
        close_kernel=5,
        min_component_area=10,
    )

    result = quantify_tumor(
        cleaned,
        coverage,
        tissue_mask=tissue,
        region_min_area=10,
    )

    Image.fromarray(
        raw,
        mode="L",
    ).save(
        output / "raw_mask.png"
    )

    Image.fromarray(
        cleaned,
        mode="L",
    ).save(
        output / "cleaned_mask.png"
    )

    print("Segmentation synthetic demo")
    print(
        f"Tumor % of analyzable area: "
        f"{result['tumor_percentage_of_analyzable_area']:.2f}%"
    )
    print(
        f"Region count: {result['region_count']}"
    )
    print(
        f"Largest region: "
        f"{result['largest_region_pixels_thumbnail']}"
    )
    print(
        f"Artifacts: {output}"
    )


if __name__ == "__main__":
    main()

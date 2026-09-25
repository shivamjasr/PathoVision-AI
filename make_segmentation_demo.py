import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic image/mask pairs for a U-Net smoke test."
    )
    parser.add_argument(
        "--output",
        default="data/segmentation_demo",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--val",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output = Path(args.output)

    rows = []

    for split, count in [
        ("train", args.train),
        ("val", args.val),
    ]:
        image_dir = output / "images" / split
        mask_dir = output / "masks" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for index in range(count):
            h = w = args.size

            # Purple/pink textured background resembling a generic
            # stain-style image; this is synthetic only.
            image = np.zeros((h, w, 3), dtype=np.float32)
            image[..., 0] = rng.normal(218, 10, (h, w))
            image[..., 1] = rng.normal(178, 10, (h, w))
            image[..., 2] = rng.normal(199, 10, (h, w))
            image = np.clip(image, 0, 255).astype(np.uint8)

            mask = np.zeros(
                (h, w),
                dtype=np.uint8,
            )

            # Draw 1–3 synthetic lesions.
            lesions = int(rng.integers(1, 4))

            for _ in range(lesions):
                cx = int(rng.integers(w * 0.2, w * 0.8))
                cy = int(rng.integers(h * 0.2, h * 0.8))
                rx = int(rng.integers(w * 0.06, w * 0.18))
                ry = int(rng.integers(h * 0.06, h * 0.18))

                cv2.ellipse(
                    mask,
                    (cx, cy),
                    (rx, ry),
                    float(rng.integers(0, 180)),
                    0,
                    360,
                    255,
                    -1,
                )

            # Make tumor areas visually distinct in the synthetic image.
            image[mask > 0] = np.array(
                [120, 90, 140],
                dtype=np.uint8,
            )

            name = f"demo_{index:05d}.png"
            image_path = image_dir / name
            mask_path = mask_dir / name

            Image.fromarray(
                image,
                mode="RGB",
            ).save(image_path)

            Image.fromarray(
                mask,
                mode="L",
            ).save(mask_path)

            rows.append(
                {
                    "split": split,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "slide_id": f"synthetic_{split}",
                    "x": 0,
                    "y": 0,
                    "tumor_fraction": f"{float((mask > 0).mean()):.6f}",
                }
            )

    manifest = output / "manifest.csv"

    with manifest.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "split",
            "image_path",
            "mask_path",
            "slide_id",
            "x",
            "y",
            "tumor_fraction",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created synthetic segmentation dataset: {output}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()

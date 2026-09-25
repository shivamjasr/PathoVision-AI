import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(
        description="Create a synthetic pathology-like RGB image for pipeline tests."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/demo_slide.png",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
    )
    args = parser.parse_args()

    size = args.size
    if size < 512:
        raise ValueError("size must be at least 512.")

    # Start with a bright slide background.
    array = np.full(
        (size, size, 3),
        fill_value=250,
        dtype=np.uint8,
    )

    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image, "RGB")

    # Add tissue-like regions with slightly different purple/pink tones.
    shapes = [
        (120, 120, size * 0.55, size * 0.42, (218, 175, 196)),
        (size * 0.38, size * 0.20, size * 0.90, size * 0.58, (205, 165, 190)),
        (size * 0.12, size * 0.58, size * 0.68, size * 0.91, (226, 182, 198)),
    ]

    for x0, y0, x1, y1, fill in shapes:
        draw.ellipse(
            (int(x0), int(y0), int(x1), int(y1)),
            fill=fill,
        )

    # Add darker nuclei-like dots to make tissue less uniform.
    rng = np.random.default_rng(42)

    for _ in range(1400):
        x = int(rng.integers(70, size - 70))
        y = int(rng.integers(70, size - 70))

        if array[int(y), int(x), 0] < 245:
            radius = int(rng.integers(1, 4))
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(110, 70, 120),
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)

    print(f"Created demo image: {output}")

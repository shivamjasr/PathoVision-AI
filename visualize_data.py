import argparse
import math

import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASS_NAMES, PLOTS_DIR, make_output_dirs
from src.datasets.pcam import get_pcam_dataset


def denormalize(image: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    return np.clip(image, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    make_output_dirs()
    dataset = get_pcam_dataset("train", download=False)

    n = min(args.samples, len(dataset))
    cols = 4
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        image = denormalize(image)

        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label} ({CLASS_NAMES[int(label)]})")
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("PatchCamelyon training samples", fontsize=14)
    fig.tight_layout()

    output = PLOTS_DIR / "pcam_samples.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

import argparse
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASS_NAMES, PLOTS_DIR, make_output_dirs
from src.datasets.pcam import get_pcam_dataset


def inspect_split(split: str, dataset, max_samples: int):
    print(f"\n===== {split.upper()} =====")
    print(f"Number of samples: {len(dataset):,}")

    if len(dataset) == 0:
        return

    # We intentionally inspect only a bounded number of labels so that
    # this script stays lightweight on CPU and does not scan the full
    # dataset when we are simply checking the pipeline.
    n = min(max_samples, len(dataset))
    counts = Counter()

    for i in range(n):
        _, label = dataset[i]
        counts[int(label)] += 1

    print(f"Inspected labels: {n:,}")

    for class_id in sorted(CLASS_NAMES):
        count = counts[class_id]
        percentage = 100.0 * count / max(n, 1)
        print(
            f"{class_id} ({CLASS_NAMES[class_id]:>6}): "
            f"{count:>6}  ({percentage:6.2f}%)"
        )

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PCam dataset size, tensor format and approximate label balance."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Maximum labels to inspect per split.",
    )
    args = parser.parse_args()

    make_output_dirs()

    train = get_pcam_dataset("train", download=False)
    val = get_pcam_dataset("val", download=False)
    test = get_pcam_dataset("test", download=False)

    counts = {}
    for split, dataset in [("train", train), ("val", val), ("test", test)]:
        counts[split] = inspect_split(split, dataset, args.samples)

    # Check one raw transformed sample.
    image, label = train[0]

    print("\n===== SAMPLE CHECK =====")
    print(f"Image type : {type(image).__name__}")
    print(f"Image shape: {tuple(image.shape)}")
    print(f"Image dtype: {image.dtype}")
    print(f"Label      : {int(label)} ({CLASS_NAMES[int(label)]})")

    expected_shape = (3, 96, 96)
    if tuple(image.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected transformed image shape {tuple(image.shape)}; "
            f"expected {expected_shape}."
        )

    # Save a compact label-balance plot for the inspected labels.
    split_names = list(counts.keys())
    normal_counts = [counts[s][0] for s in split_names]
    tumor_counts = [counts[s][1] for s in split_names]

    x = np.arange(len(split_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, normal_counts, width, label="Normal")
    ax.bar(x + width / 2, tumor_counts, width, label="Tumor")
    ax.set_xticks(x, split_names)
    ax.set_ylabel("Inspected samples")
    ax.set_title("PCam approximate label distribution")
    ax.legend()

    output = PLOTS_DIR / "label_distribution.png"
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Label distribution plot: {output}")
    print("\nDataset inspection passed.")


if __name__ == "__main__":
    main()

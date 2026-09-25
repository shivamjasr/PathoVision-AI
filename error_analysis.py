import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.pcam import get_pcam_dataset


def denormalize(tensor):
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )[:, None, None]

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )[:, None, None]

    image = tensor.numpy() * std + mean

    return np.clip(
        np.transpose(
            image,
            (1, 2, 0),
        ),
        0,
        1,
    )


def save_contact_sheet(
    dataset,
    indices,
    probabilities,
    labels,
    title,
    output_path,
):
    if len(indices) == 0:
        print(
            f"No examples for {title}."
        )
        return

    count = len(indices)
    cols = 5
    rows = int(
        np.ceil(count / cols)
    )

    fig = plt.figure(
        figsize=(15, 3 * rows)
    )

    for position, index in enumerate(
        indices
    ):
        image, label = dataset[int(index)]
        image = denormalize(image)

        ax = fig.add_subplot(
            rows,
            cols,
            position + 1,
        )

        ax.imshow(image)
        ax.set_title(
            f"true={int(label)} "
            f"p={probabilities[int(index)]:.3f}"
        )
        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=14,
    )
    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Find false positives, false negatives and uncertain examples."
    )
    parser.add_argument(
        "--predictions",
        required=True,
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--output",
        default="results/experiments/errors",
    )
    args = parser.parse_args()

    data = np.load(
        args.predictions
    )

    labels = data["labels"].astype(
        np.int64
    )
    probabilities = data["probabilities"].astype(
        np.float32
    )

    predictions = (
        probabilities >= args.threshold
    ).astype(np.int64)

    false_positive = np.where(
        (predictions == 1)
        & (labels == 0)
    )[0]

    false_negative = np.where(
        (predictions == 0)
        & (labels == 1)
    )[0]

    uncertainty = np.argsort(
        np.abs(
            probabilities - 0.5
        )
    )[:args.top]

    # Hard errors first by confidence:
    # FP with highest p and FN with lowest p.
    fp_sorted = false_positive[
        np.argsort(
            probabilities[
                false_positive
            ]
        )[::-1]
    ][:args.top]

    fn_sorted = false_negative[
        np.argsort(
            probabilities[
                false_negative
            ]
        )
    ][:args.top]

    dataset = get_pcam_dataset(
        args.split,
        download=False,
        image_size=96,
    )

    output = Path(
        args.output
    )

    save_contact_sheet(
        dataset,
        fp_sorted,
        probabilities,
        labels,
        "False positives",
        output / "false_positives.png",
    )

    save_contact_sheet(
        dataset,
        fn_sorted,
        probabilities,
        labels,
        "False negatives",
        output / "false_negatives.png",
    )

    save_contact_sheet(
        dataset,
        uncertainty,
        probabilities,
        labels,
        "Most uncertain examples",
        output / "uncertain_examples.png",
    )

    csv_path = output / "errors.csv"
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "index",
                "label",
                "probability",
                "prediction",
                "error_type",
            ]
        )

        for index in fp_sorted:
            writer.writerow(
                [
                    int(index),
                    int(labels[index]),
                    float(probabilities[index]),
                    int(predictions[index]),
                    "false_positive",
                ]
            )

        for index in fn_sorted:
            writer.writerow(
                [
                    int(index),
                    int(labels[index]),
                    float(probabilities[index]),
                    int(predictions[index]),
                    "false_negative",
                ]
            )

    print(
        f"False positives found: {len(false_positive):,}"
    )
    print(
        f"False negatives found: {len(false_negative):,}"
    )
    print(
        f"Uncertain examples saved: {len(uncertainty):,}"
    )
    print(
        f"CSV: {csv_path}"
    )


if __name__ == "__main__":
    main()

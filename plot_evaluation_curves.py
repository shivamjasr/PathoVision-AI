import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    roc_auc_score,
    average_precision_score,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions",
        type=str,
    )
    parser.add_argument(
        "--name",
        default="model",
    )
    parser.add_argument(
        "--output",
        default="results/experiments",
    )
    args = parser.parse_args()

    data = np.load(args.predictions)

    y_true = data["labels"].astype(np.int64)
    probabilities = data["probabilities"].astype(
        np.float32
    )

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "ROC/PR curves require both classes."
        )

    output = Path(
        args.output
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    ap = average_precision_score(
        y_true,
        probabilities,
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )
    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        name=f"{args.name} (AUC={roc_auc:.3f})",
        ax=ax,
    )
    fig.tight_layout()

    roc_path = (
        output
        / f"{args.name}_roc_curve.png"
    )
    fig.savefig(
        roc_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )
    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        name=f"{args.name} (AP={ap:.3f})",
        ax=ax,
    )
    fig.tight_layout()

    pr_path = (
        output
        / f"{args.name}_pr_curve.png"
    )
    fig.savefig(
        pr_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )
    print(
        f"Average precision: {ap:.4f}"
    )
    print(f"ROC curve: {roc_path}")
    print(f"PR curve : {pr_path}")


if __name__ == "__main__":
    main()

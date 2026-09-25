import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


def metrics_at_threshold(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Choose a probability threshold on validation data. "
            "The test set must not be used for threshold selection."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
    )
    parser.add_argument(
        "--output",
        default="results/experiments/threshold_selection.json",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
    )
    args = parser.parse_args()

    data = np.load(
        args.predictions
    )

    y_true = data["labels"].astype(
        np.int64
    )
    probabilities = data["probabilities"].astype(
        np.float32
    )

    thresholds = np.arange(
        args.start,
        args.stop + args.step / 2,
        args.step,
    )

    results = [
        metrics_at_threshold(
            y_true,
            probabilities,
            float(threshold),
        )
        for threshold in thresholds
    ]

    best = max(
        results,
        key=lambda item: (
            item["f1"],
            item["recall"],
        ),
    )

    output = Path(
        args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "source_predictions": str(
            Path(args.predictions).resolve()
        ),
        "selection_metric": "f1",
        "best_threshold": best,
        "all_thresholds": results,
        "warning": (
            "Threshold selected on validation data only. "
            "Do not optimize it against the final test set."
        ),
    }

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"Best validation threshold: "
        f"{best['threshold']:.2f}"
    )
    print(
        f"Precision: {best['precision']:.4f}"
    )
    print(
        f"Recall   : {best['recall']:.4f}"
    )
    print(
        f"F1       : {best['f1']:.4f}"
    )
    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()

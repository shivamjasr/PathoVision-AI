import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="checkpoints/hybrid_models/rf_handcrafted.joblib",
    )
    parser.add_argument(
        "--names",
        default=(
            "data/hybrid/features/"
            "handcrafted_feature_names.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/experiments/handcrafted_feature_importance.png",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
    )
    args = parser.parse_args()

    model = joblib.load(
        args.model
    )

    names = json.loads(
        Path(args.names).read_text(
            encoding="utf-8"
        )
    )

    importances = np.asarray(
        model.feature_importances_
    )

    if len(names) != len(importances):
        raise ValueError(
            "Feature names and importances have different lengths."
        )

    order = np.argsort(
        importances
    )[::-1][:args.top]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    y_positions = np.arange(
        len(order)
    )

    ax.barh(
        y_positions,
        importances[order][::-1],
    )
    ax.set_yticks(
        y_positions,
        [names[i] for i in order][::-1],
    )
    ax.set_xlabel("Random Forest impurity importance")
    ax.set_title(
        "Top handcrafted feature importances"
    )

    fig.tight_layout()

    output = Path(
        args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

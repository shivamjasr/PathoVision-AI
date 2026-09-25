import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_feature_data(directory):
    directory = Path(directory)

    data = {}

    for split in ("train", "val", "test"):
        data[f"{split}_h"] = np.load(
            directory
            / f"{split}_handcrafted.npy"
        )
        data[f"{split}_d"] = np.load(
            directory
            / f"{split}_deep.npy"
        )
        data[f"{split}_y"] = np.load(
            directory
            / f"{split}_labels.npy"
        )

    names_path = (
        directory
        / "handcrafted_feature_names.json"
    )

    names = json.loads(
        names_path.read_text(
            encoding="utf-8"
        )
    )

    if len(names) != data["train_h"].shape[1]:
        raise ValueError(
            "Feature-name count does not match matrix width."
        )

    return data, names


def evaluate(y_true, probability):
    prediction = (
        probability >= 0.5
    ).astype(np.int64)

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                prediction,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),
    }


def train_eval(
    name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    seed,
):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
    )

    model.fit(
        X_train,
        y_train,
    )

    val_probability = model.predict_proba(
        X_val
    )[:, 1]

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    return {
        "name": name,
        "validation": evaluate(
            y_val,
            val_probability,
        ),
        "test": evaluate(
            y_test,
            test_probability,
        ),
    }, model


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure which feature groups contribute "
            "to the classical/hybrid model."
        )
    )
    parser.add_argument(
        "--features",
        default="data/hybrid/features",
    )
    parser.add_argument(
        "--output",
        default="results/experiments/ablation_results.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    directory = Path(
        args.features
    )

    data, names = load_feature_data(
        directory
    )

    # We keep the feature-name mapping explicit:
    # RGB/HSV = 24 each? Actually each color-space has
    # 3 channels × 4 statistics = 12.
    # GLCM = 8, edge = 1, morphology = 16,
    # channel ratios = 12.
    #
    # Use prefixes for robust grouping rather than hardcoded positions.
    color_idx = [
        i for i, name in enumerate(names)
        if name.startswith(("RGB_", "HSV_"))
    ]

    texture_idx = [
        i for i, name in enumerate(names)
        if name.startswith("GLCM_")
    ] + [
        i for i, name in enumerate(names)
        if name == "edge_density"
    ]

    morphology_idx = [
        i for i, name in enumerate(names)
        if name.startswith("morph_")
    ]

    ratio_idx = [
        i for i, name in enumerate(names)
        if name.startswith("ratio_")
    ]

    handcrafted_idx = (
        color_idx
        + texture_idx
        + morphology_idx
        + ratio_idx
    )

    if len(handcrafted_idx) != len(names):
        raise ValueError(
            "Feature grouping did not cover every handcrafted feature."
        )

    y_train = data["train_y"]
    y_val = data["val_y"]
    y_test = data["test_y"]

    experiments = []

    groups = {
        "color_only": color_idx,
        "texture_only": texture_idx,
        "morphology_only": morphology_idx,
        "ratios_only": ratio_idx,
        "all_handcrafted": handcrafted_idx,
    }

    for group_name, indices in groups.items():
        result, _ = train_eval(
            group_name,
            data["train_h"][:, indices],
            y_train,
            data["val_h"][:, indices],
            y_val,
            data["test_h"][:, indices],
            y_test,
            args.seed,
        )
        result["feature_count"] = len(indices)
        experiments.append(result)

    # Deep-only and fused baselines.
    result, _ = train_eval(
        "deep_only_random_forest",
        data["train_d"],
        y_train,
        data["val_d"],
        y_val,
        data["test_d"],
        y_test,
        args.seed,
    )
    result["feature_count"] = int(
        data["train_d"].shape[1]
    )
    experiments.append(result)

    train_fused = np.concatenate(
        [data["train_h"], data["train_d"]],
        axis=1,
    )
    val_fused = np.concatenate(
        [data["val_h"], data["val_d"]],
        axis=1,
    )
    test_fused = np.concatenate(
        [data["test_h"], data["test_d"]],
        axis=1,
    )

    result, _ = train_eval(
        "fused_all",
        train_fused,
        y_train,
        val_fused,
        y_val,
        test_fused,
        y_test,
        args.seed,
    )
    result["feature_count"] = int(
        train_fused.shape[1]
    )
    experiments.append(result)

    output = Path(
        args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "experiments": experiments,
        "feature_groups": {
            "color": [
                names[i] for i in color_idx
            ],
            "texture": [
                names[i] for i in texture_idx
            ],
            "morphology": [
                names[i] for i in morphology_idx
            ],
            "ratios": [
                names[i] for i in ratio_idx
            ],
        },
    }

    output.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nAblation study")
    print("-" * 76)

    for result in experiments:
        print(
            f"{result['name']:<28} "
            f"features={result['feature_count']:<5} "
            f"val_auc={result['validation']['roc_auc']:.4f} "
            f"test_auc={result['test']['roc_auc']:.4f}"
        )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()

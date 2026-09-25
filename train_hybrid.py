import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def load_split(directory, split):
    directory = Path(directory)

    handcrafted = np.load(
        directory
        / f"{split}_handcrafted.npy"
    )

    deep = np.load(
        directory
        / f"{split}_deep.npy"
    )

    labels = np.load(
        directory
        / f"{split}_labels.npy"
    )

    return handcrafted, deep, labels


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


def fit_and_score(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
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

    return (
        model,
        {
            "model": name,
            "validation": evaluate(
                y_val,
                val_probability,
            ),
            "test": evaluate(
                y_test,
                test_probability,
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train classical and hybrid ML models."
    )
    parser.add_argument(
        "--features",
        default="data/hybrid/features",
    )
    parser.add_argument(
        "--models",
        default="checkpoints/hybrid_models",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    features_dir = Path(
        args.features
    )
    model_dir = Path(
        args.models
    )
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_h, train_d, y_train = load_split(
        features_dir,
        "train",
    )
    val_h, val_d, y_val = load_split(
        features_dir,
        "val",
    )
    test_h, test_d, y_test = load_split(
        features_dir,
        "test",
    )

    print(
        f"Train={len(y_train):,}, "
        f"Val={len(y_val):,}, "
        f"Test={len(y_test):,}"
    )

    # ---------------------------------------------------------------
    # Model A: classical ML using handcrafted features.
    # ---------------------------------------------------------------
    handcrafted_model = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
    )

    handcrafted_model, handcrafted_result = (
        fit_and_score(
            "random_forest_handcrafted",
            handcrafted_model,
            train_h,
            y_train,
            val_h,
            y_val,
            test_h,
            y_test,
        )
    )

    # ---------------------------------------------------------------
    # Model B: ML over ResNet18's learned representation.
    # Scaling is fit ONLY on train to avoid leakage.
    # ---------------------------------------------------------------
    scaler = StandardScaler()
    train_d_scaled = scaler.fit_transform(
        train_d
    )
    val_d_scaled = scaler.transform(
        val_d
    )
    test_d_scaled = scaler.transform(
        test_d
    )

    deep_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=args.seed,
    )

    deep_model, deep_result = fit_and_score(
        "logistic_regression_resnet_features",
        deep_model,
        train_d_scaled,
        y_train,
        val_d_scaled,
        y_val,
        test_d_scaled,
        y_test,
    )

    # ---------------------------------------------------------------
    # Model C: Random Forest on fused classical + deep features.
    # ---------------------------------------------------------------
    train_fused = np.concatenate(
        [train_h, train_d],
        axis=1,
    )
    val_fused = np.concatenate(
        [val_h, val_d],
        axis=1,
    )
    test_fused = np.concatenate(
        [test_h, test_d],
        axis=1,
    )

    fused_model = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
    )

    fused_model, fused_result = fit_and_score(
        "random_forest_fused",
        fused_model,
        train_fused,
        y_train,
        val_fused,
        y_val,
        test_fused,
        y_test,
    )

    results = [
        handcrafted_result,
        deep_result,
        fused_result,
    ]

    print("\nValidation/Test summary")
    print("-" * 74)

    for result in results:
        val = result["validation"]
        test = result["test"]

        print(
            f"{result['model']:<38} "
            f"val_auc={val['roc_auc']:.4f} "
            f"test_auc={test['roc_auc']:.4f} "
            f"test_f1={test['f1']:.4f}"
        )

    joblib.dump(
        handcrafted_model,
        model_dir
        / "rf_handcrafted.joblib",
    )
    joblib.dump(
        deep_model,
        model_dir
        / "lr_resnet_features.joblib",
    )
    joblib.dump(
        scaler,
        model_dir
        / "resnet_scaler.joblib",
    )
    joblib.dump(
        fused_model,
        model_dir
        / "rf_fused.joblib",
    )

    report = {
        "feature_dimensions": {
            "handcrafted": int(
                train_h.shape[1]
            ),
            "deep": int(
                train_d.shape[1]
            ),
            "fused": int(
                train_fused.shape[1]
            ),
        },
        "results": results,
        "random_state": args.seed,
    }

    report_path = (
        model_dir
        / "hybrid_model_results.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved models and report to: "
        f"{model_dir}"
    )


if __name__ == "__main__":
    main()

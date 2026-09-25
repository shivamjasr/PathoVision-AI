import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(
        description="Compare held-out metrics from baseline CNN and ResNet18."
    )
    parser.add_argument(
        "--baseline",
        default="results/metrics/test_metrics.json",
    )
    parser.add_argument(
        "--resnet",
        default="results/metrics/resnet18_test_metrics.json",
    )
    args = parser.parse_args()

    baseline = load_json(Path(args.baseline))
    resnet = load_json(Path(args.resnet))

    if baseline is None:
        raise FileNotFoundError(
            f"Baseline metrics not found: {args.baseline}"
        )

    if resnet is None:
        raise FileNotFoundError(
            f"ResNet metrics not found: {args.resnet}"
        )

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    print("\nModel comparison")
    print("-" * 54)
    print(f"{'Metric':<14}{'Baseline CNN':>18}{'ResNet18':>18}")
    print("-" * 54)

    for metric in metrics:
        base = baseline.get(metric)
        res = resnet.get(metric)

        base_text = "N/A" if base is None else f"{base:.4f}"
        res_text = "N/A" if res is None else f"{res:.4f}"

        print(f"{metric:<14}{base_text:>18}{res_text:>18}")

    print("-" * 54)
    print("Do not choose a model from one metric alone.")
    print("Use the validation/test protocol plus the intended error trade-offs.")


if __name__ == "__main__":
    main()

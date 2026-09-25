import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Print a concise hybrid model comparison."
    )
    parser.add_argument(
        "--report",
        default=(
            "checkpoints/hybrid_models/"
            "hybrid_model_results.json"
        ),
    )
    args = parser.parse_args()

    path = Path(args.report)

    if not path.exists():
        raise FileNotFoundError(
            f"Hybrid report not found: {path}"
        )

    report = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    print("\nHybrid model comparison")
    print("-" * 74)
    print(
        f"{'Model':<40}"
        f"{'Val AUC':>12}"
        f"{'Test AUC':>12}"
        f"{'Test F1':>10}"
    )
    print("-" * 74)

    for result in report["results"]:
        print(
            f"{result['model']:<40}"
            f"{result['validation']['roc_auc']:>12.4f}"
            f"{result['test']['roc_auc']:>12.4f}"
            f"{result['test']['f1']:>10.4f}"
        )

    print("-" * 74)
    print(
        "The comparison is descriptive; choose the final model "
        "using validation evidence, error costs, compute and the "
        "downstream WSI application."
    )


if __name__ == "__main__":
    main()

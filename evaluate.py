import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    get_device,
    make_output_dirs,
)
from src.datasets.pcam import get_pcam_dataset
from src.models.cnn import TumorCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-test", type=int, default=None)
    args = parser.parse_args()

    make_output_dirs()
    device = get_device()
    print(f"Using device: {device}")

    dataset = get_pcam_dataset("test", download=False)

    if args.limit_test is not None and args.limit_test > 0:
        dataset = Subset(dataset, range(min(args.limit_test, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint_path = CHECKPOINT_DIR / "best_cnn.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the model first."
        )

    model = TumorCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            logits = model(images)

            y_prob.append(torch.sigmoid(logits).cpu().numpy())
            y_true.append(labels.numpy())

    y_true = np.concatenate(y_true).astype(int)
    y_prob = np.concatenate(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else None
        ),
    }

    print("\nTest metrics")
    for name, value in metrics.items():
        print(f"{name:>10}: {value}")

    metrics_path = METRICS_DIR / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Tumor"],
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, values_format="d")
    ax.set_title("PCam Test Confusion Matrix")
    fig.tight_layout()

    plot_path = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nMetrics: {metrics_path}")
    print(f"Plot:    {plot_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LEARNING_RATE,
    METRICS_DIR,
    PLOTS_DIR,
    WEIGHT_DECAY,
    get_device,
    make_output_dirs,
)
from src.datasets.pcam import make_loaders
from src.models.cnn import TumorCNN


def set_seed(seed: int) -> None:
    """Make common random operations reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else None
        ),
    }

    return result


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training: bool,
):
    """Run one complete training or validation epoch."""
    model.train(training)

    total_loss = 0.0
    targets = []
    probabilities = []

    description = "train" if training else "val"
    iterator = tqdm(loader, desc=description, leave=False)

    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Keeps a bad gradient step from destabilizing training.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()

        probs = torch.sigmoid(logits)

        total_loss += loss.item() * images.size(0)
        targets.append(labels.detach().cpu().numpy())
        probabilities.append(probs.detach().cpu().numpy())

        iterator.set_postfix(loss=f"{loss.item():.4f}")

    y_true = np.concatenate(targets)
    y_prob = np.concatenate(probabilities)

    average_loss = total_loss / len(loader.dataset)
    metrics = binary_metrics(y_true, y_prob)

    return average_loss, metrics


def save_training_curves(history):
    """Save loss, F1 and ROC-AUC curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig = plt.figure(figsize=(15, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set_title("Binary Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(epochs, history["train_f1"], label="train")
    ax2.plot(epochs, history["val_f1"], label="validation")
    ax2.set_title("F1 Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(epochs, history["train_auc"], label="train")
    ax3.plot(epochs, history["val_auc"], label="validation")
    ax3.set_title("ROC-AUC")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("AUC")
    ax3.legend()

    fig.tight_layout()

    output = PLOTS_DIR / "training_curves.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Train the baseline CNN on PCam."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=1)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_output_dirs()
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _ = make_loaders(
        download=False,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = TumorCNN().to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Raw logits from TumorCNN + BCEWithLogitsLoss is more numerically
    # stable than applying sigmoid before the loss.
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1": [],
        "val_f1": [],
        "train_auc": [],
        "val_auc": [],
        "learning_rate": [],
    }

    best_score = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            training=False,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        for metric_name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
        ):
            history[f"train_{metric_name}"].append(
                train_metrics[metric_name]
            )
            history[f"val_{metric_name}"].append(
                val_metrics[metric_name]
            )

        history["learning_rate"].append(current_lr)

        # Prefer ROC-AUC because it does not depend on the 0.5 threshold.
        score = val_metrics["roc_auc"]
        if score is None:
            score = val_metrics["f1"]

        scheduler.step(score)

        print(
            f"Train | loss={train_loss:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"prec={train_metrics['precision']:.4f} "
            f"recall={train_metrics['recall']:.4f} "
            f"f1={train_metrics['f1']:.4f} "
            f"auc={train_metrics['roc_auc']}"
        )
        print(
            f"Val   | loss={val_loss:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"prec={val_metrics['precision']:.4f} "
            f"recall={val_metrics['recall']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"auc={val_metrics['roc_auc']}"
        )
        print(f"LR    | {current_lr:.6g}")

        # Always save the most recent checkpoint so training can be inspected.
        latest_path = CHECKPOINT_DIR / "latest_cnn.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_score": score,
                "history": history,
            },
            latest_path,
        )

        if score > best_score:
            best_score = score
            epochs_without_improvement = 0

            best_path = CHECKPOINT_DIR / "best_cnn.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_score": score,
                    "history": history,
                },
                best_path,
            )
            print(f"Saved best checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement "
                f"({epochs_without_improvement}/{args.patience})"
            )

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    history_path = METRICS_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    curve_path = save_training_curves(history)

    summary = {
        "epochs_requested": args.epochs,
        "epochs_completed": len(history["train_loss"]),
        "best_validation_score": best_score,
        "best_checkpoint": str(CHECKPOINT_DIR / "best_cnn.pt"),
        "device": str(device),
        "trainable_parameters": count_parameters(model),
        "seed": args.seed,
    }

    summary_path = METRICS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nTraining complete.")
    print(f"History : {history_path}")
    print(f"Summary : {summary_path}")
    print(f"Curves  : {curve_path}")


if __name__ == "__main__":
    main()

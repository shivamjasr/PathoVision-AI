import argparse
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.config import (
    CHECKPOINT_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    get_device,
    make_output_dirs,
)
from src.segmentation.dataset import SegmentationDataset
from src.segmentation.losses import (
    CombinedSegmentationLoss,
    segmentation_metrics,
)
from src.models.unet import UNet
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training,
):
    model.train(training)

    total_loss = 0.0
    metric_sums = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    batches = 0

    description = "train" if training else "val"

    for batch in tqdm(
        loader,
        desc=description,
        leave=False,
    ):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        metrics = segmentation_metrics(
            logits,
            masks,
        )

        for key in metric_sums:
            metric_sums[key] += metrics[key]

        batches += 1

    average_loss = (
        total_loss / len(loader.dataset)
    )

    metrics = {
        key: value / max(batches, 1)
        for key, value in metric_sums.items()
    }

    return average_loss, metrics


def save_curves(history):
    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    fig = plt.figure(figsize=(15, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(
        epochs,
        history["train_loss"],
        label="train",
    )
    ax1.plot(
        epochs,
        history["val_loss"],
        label="validation",
    )
    ax1.set_title("Segmentation Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(
        epochs,
        history["train_dice"],
        label="train",
    )
    ax2.plot(
        epochs,
        history["val_dice"],
        label="validation",
    )
    ax2.set_title("Dice Score")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(
        epochs,
        history["train_iou"],
        label="train",
    )
    ax3.plot(
        epochs,
        history["val_iou"],
        label="validation",
    )
    ax3.set_title("IoU")
    ax3.set_xlabel("Epoch")
    ax3.legend()

    fig.tight_layout()

    output = PLOTS_DIR / "unet_training_curves.png"
    fig.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Train compact U-Net for binary tumor segmentation."
    )
    parser.add_argument(
        "--manifest",
        default="data/segmentation_demo/manifest.csv",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    make_output_dirs()
    set_seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    train_dataset = SegmentationDataset(
        args.manifest,
        split="train",
        image_size=args.image_size,
        augment=True,
    )
    val_dataset = SegmentationDataset(
        args.manifest,
        split="val",
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    criterion = CombinedSegmentationLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=args.pos_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "learning_rate": [],
    }

    best_dice = -1.0
    stagnant_epochs = 0

    print(
        f"Train patches: {len(train_dataset):,}"
    )
    print(
        f"Val patches  : {len(val_dataset):,}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        print(
            f"\nEpoch {epoch}/{args.epochs}"
        )

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

        scheduler.step(
            val_metrics["dice"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        for name in (
            "dice",
            "iou",
            "precision",
            "recall",
        ):
            history[f"train_{name}"].append(
                train_metrics[name]
            )
            history[f"val_{name}"].append(
                val_metrics[name]
            )

        history["learning_rate"].append(
            current_lr
        )

        print(
            f"Train | loss={train_loss:.4f} "
            f"dice={train_metrics['dice']:.4f} "
            f"iou={train_metrics['iou']:.4f}"
        )
        print(
            f"Val   | loss={val_loss:.4f} "
            f"dice={val_metrics['dice']:.4f} "
            f"iou={val_metrics['iou']:.4f} "
            f"precision={val_metrics['precision']:.4f} "
            f"recall={val_metrics['recall']:.4f}"
        )
        print(
            f"LR    | {current_lr:.6g}"
        )

        latest_path = (
            CHECKPOINT_DIR
            / "latest_unet.pt"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_dice": val_metrics["dice"],
                "image_size": args.image_size,
                "base_channels": args.base_channels,
                "history": history,
            },
            latest_path,
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            stagnant_epochs = 0

            best_path = (
                CHECKPOINT_DIR
                / "best_unet.pt"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_dice": best_dice,
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                    "history": history,
                },
                best_path,
            )

            print(
                f"Saved best checkpoint: {best_path}"
            )
        else:
            stagnant_epochs += 1
            print(
                f"No Dice improvement "
                f"({stagnant_epochs}/{args.patience})"
            )

        if stagnant_epochs >= args.patience:
            print("Early stopping triggered.")
            break

    history_path = (
        METRICS_DIR
        / "unet_training_history.json"
    )
    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    curve_path = save_curves(history)

    print("\nU-Net training complete.")
    print(f"History: {history_path}")
    print(f"Curves : {curve_path}")


if __name__ == "__main__":
    main()

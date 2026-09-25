import torch
import torch.nn as nn


def soft_dice_score(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(1)
    targets = targets.flatten(1)

    intersection = (probabilities * targets).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        (2.0 * intersection + smooth)
        / (denominator + smooth)
    )

    return dice.mean()


class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        return 1.0 - soft_dice_score(
            logits,
            targets,
        )


class CombinedSegmentationLoss(nn.Module):
    """BCE-with-logits + Dice loss for binary segmentation."""

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        pos_weight=None,
    ):
        super().__init__()

        if pos_weight is not None:
            self.register_buffer(
                "pos_weight_tensor",
                torch.tensor(
                    [float(pos_weight)],
                    dtype=torch.float32,
                ),
            )
        else:
            self.pos_weight_tensor = None

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits, targets):
        pos_weight = self.pos_weight_tensor
        if pos_weight is not None:
            pos_weight = pos_weight.to(
                device=logits.device,
                dtype=logits.dtype,
            )

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
        )

        dice = 1.0 - soft_dice_score(
            logits,
            targets,
        )

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
        )


def segmentation_metrics(
    logits,
    targets,
    threshold=0.5,
    eps=1e-7,
):
    predictions = (
        torch.sigmoid(logits)
        >= threshold
    )

    targets_bool = targets >= 0.5

    predictions = predictions.flatten(1)
    targets_bool = targets_bool.flatten(1)

    tp = (
        predictions
        & targets_bool
    ).sum(dim=1).float()

    fp = (
        predictions
        & ~targets_bool
    ).sum(dim=1).float()

    fn = (
        ~predictions
        & targets_bool
    ).sum(dim=1).float()

    dice = (
        (2 * tp + eps)
        / (2 * tp + fp + fn + eps)
    ).mean()

    iou = (
        (tp + eps)
        / (tp + fp + fn + eps)
    ).mean()

    precision = (
        (tp + eps)
        / (tp + fp + eps)
    ).mean()

    recall = (
        (tp + eps)
        / (tp + fn + eps)
    ).mean()

    return {
        "dice": float(dice.item()),
        "iou": float(iou.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
    }

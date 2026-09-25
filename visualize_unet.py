import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import CHECKPOINT_DIR, PLOTS_DIR, get_device, make_output_dirs
from src.models.unet import UNet
from src.segmentation.dataset import SegmentationDataset


def denormalize(tensor):
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=tensor.device,
    )[:, None, None]

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=tensor.device,
    )[:, None, None]

    image = tensor * std + mean
    return image.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/segmentation_demo/manifest.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            CHECKPOINT_DIR / "best_unet.pt"
        ),
    )
    parser.add_argument(
        "--split",
        default="val",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    make_output_dirs()
    device = get_device()

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    image_size = checkpoint.get(
        "image_size",
        128,
    )
    base_channels = checkpoint.get(
        "base_channels",
        16,
    )

    dataset = SegmentationDataset(
        args.manifest,
        split=args.split,
        image_size=image_size,
        augment=False,
    )

    index = (
        args.index
        if args.index >= 0
        else random.randrange(len(dataset))
    )

    sample = dataset[index]

    model = UNet(
        base_channels=base_channels
    ).to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    with torch.no_grad():
        image_tensor = sample["image"].unsqueeze(0).to(device)
        logits = model(image_tensor)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

    image = denormalize(
        sample["image"]
    ).permute(1, 2, 0).cpu().numpy()

    ground_truth = (
        sample["mask"][0].numpy()
    )

    prediction = (
        probability >= args.threshold
    ).astype(np.float32)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Input patch")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(ground_truth, vmin=0, vmax=1)
    ax2.set_title("Ground-truth mask")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(prediction, vmin=0, vmax=1)
    ax3.set_title("U-Net prediction")
    ax3.axis("off")

    fig.suptitle(
        f"Segmentation example — {sample['slide_id']} "
        f"@ ({sample['x']}, {sample['y']})"
    )
    fig.tight_layout()

    output = (
        PLOTS_DIR
        / "unet_prediction_example.png"
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

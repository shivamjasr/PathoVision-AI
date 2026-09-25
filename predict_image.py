import argparse

import torch
from PIL import Image
from torchvision import transforms

from src.config import CHECKPOINT_DIR, get_device
from src.datasets.pcam import MEAN, STD
from src.models.cnn import TumorCNN
from src.models.resnet18 import build_resnet18


def build_transform(model_name: str):
    if model_name == "cnn":
        size = 96
    elif model_name == "resnet18":
        size = 224
    else:
        raise ValueError("model must be cnn or resnet18")

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run PathoVision classifier on one image."
    )
    parser.add_argument("image", type=str)
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet18"],
        default="cnn",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    device = get_device()

    if args.checkpoint is None:
        filename = (
            "best_cnn.pt"
            if args.model == "cnn"
            else "best_resnet18.pt"
        )
        checkpoint_path = CHECKPOINT_DIR / filename
    else:
        checkpoint_path = args.checkpoint

    image_path = args.image

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find image: {image_path}"
        ) from exc

    tensor = build_transform(args.model)(image).unsqueeze(0).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if args.model == "cnn":
        model = TumorCNN().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            probability = torch.sigmoid(model(tensor)).item()

    else:
        strategy = checkpoint.get("strategy", "full")
        model = build_resnet18(
            pretrained=False,
            strategy="full",
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)
            probability = probabilities[0, 1].item()

    prediction = "tumor" if probability >= args.threshold else "normal"

    print(f"Model      : {args.model}")
    print(f"Image      : {image_path}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Probability: {probability:.4f}")
    print(f"Prediction : {prediction}")
    print(f"Threshold  : {args.threshold:.2f}")


if __name__ == "__main__":
    main()

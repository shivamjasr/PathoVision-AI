import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import CHECKPOINT_DIR, get_device
from src.datasets.pcam import MEAN, STD
from src.models.resnet18 import build_resnet18


class GradCAM:
    """Minimal Grad-CAM implementation for ResNet18 layer4[-1]."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_tensor, class_index=1):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(
            image_tensor
        )

        target = logits[
            0,
            class_index,
        ]

        target.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * self.activations
        ).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam[0].detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM for ResNet18 tumor prediction."
    )
    parser.add_argument(
        "image",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            CHECKPOINT_DIR
            / "best_resnet18.pt"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/experiments/gradcam.png",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
    )
    args = parser.parse_args()

    device = get_device()

    image_path = Path(
        args.image
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    checkpoint = Path(
        args.checkpoint
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                MEAN,
                STD,
            ),
        ]
    )

    tensor = transform(
        image
    ).unsqueeze(0).to(device)

    saved = torch.load(
        checkpoint,
        map_location=device,
    )

    model = build_resnet18(
        pretrained=False,
        strategy="full",
    ).to(device)

    model.load_state_dict(
        saved["model_state_dict"]
    )
    model.eval()

    target_layer = model.layer4[-1].conv2

    cam_engine = GradCAM(
        model,
        target_layer,
    )

    try:
        with torch.no_grad():
            logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0].detach().cpu().numpy()

        predicted_class = int(
            probabilities.argmax()
        )

        # Re-run with gradients for Grad-CAM.
        cam = cam_engine(
            tensor,
            class_index=predicted_class,
        )
    finally:
        cam_engine.close()

    resized_cam = np.asarray(
        Image.fromarray(
            (cam * 255).astype(np.uint8)
        ).resize(
            (224, 224)
        )
    ) / 255.0

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.imshow(
        image.resize((224, 224))
    )

    ax.imshow(
        resized_cam,
        alpha=args.alpha,
    )

    ax.set_title(
        f"Predicted class={predicted_class} | "
        f"tumor probability={probabilities[1]:.3f}"
    )
    ax.axis("off")

    fig.tight_layout()

    output = Path(
        args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        f"Prediction tumor probability: "
        f"{probabilities[1]:.4f}"
    )
    print(
        f"Grad-CAM output: {output}"
    )


if __name__ == "__main__":
    main()

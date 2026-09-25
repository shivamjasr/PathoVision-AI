import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from src.hybrid.features import (
    ImageFeatureExtractor,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the fused ML + DL model on one image."
    )
    parser.add_argument(
        "image",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_resnet18.pt",
    )
    parser.add_argument(
        "--models",
        default="checkpoints/hybrid_models",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    image_path = Path(
        args.image
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    models_dir = Path(
        args.models
    )

    fused_model = joblib.load(
        models_dir
        / "rf_fused.joblib"
    )

    extractor = ImageFeatureExtractor(
        checkpoint_path=args.checkpoint
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    result = extractor.extract(
        image
    )

    fused = np.concatenate(
        [
            result.handcrafted,
            result.deep,
        ]
    )[None, :]

    probability = float(
        fused_model.predict_proba(
            fused
        )[0, 1]
    )

    prediction = (
        "tumor"
        if probability >= args.threshold
        else "normal"
    )

    print(f"Image      : {image_path}")
    print(f"Model      : fused Random Forest")
    print(f"Probability: {probability:.4f}")
    print(f"Prediction : {prediction}")
    print(
        f"Threshold  : {args.threshold:.2f}"
    )


if __name__ == "__main__":
    main()

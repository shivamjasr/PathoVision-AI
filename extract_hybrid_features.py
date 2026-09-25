import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import DATA_DIR, CHECKPOINT_DIR, make_output_dirs, get_device
from src.datasets.pcam import get_pcam_dataset
from src.hybrid.features import (
    ImageFeatureExtractor,
    feature_names,
)


def process_split(
    split,
    max_samples,
    extractor,
):
    dataset = get_pcam_dataset(
        split,
        download=False,
    )

    if max_samples is not None:
        n = min(
            max_samples,
            len(dataset),
        )
    else:
        n = len(dataset)

    handcrafted = []
    deep = []
    labels = []

    iterator = tqdm(
        range(n),
        desc=f"features:{split}",
    )

    for index in iterator:
        image, label = dataset[index]

        # dataset transform returns a normalized tensor. We need the
        # original RGB image for handcrafted feature extraction.
        tensor = image.detach().cpu()

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )[:, None, None]

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )[:, None, None]

        rgb = (
            tensor.numpy() * std + mean
        ).clip(0, 1)

        rgb = (
            np.transpose(
                rgb,
                (1, 2, 0),
            )
            * 255
        ).astype(np.uint8)

        pil_image = Image.fromarray(
            rgb,
            mode="RGB",
        )

        result = extractor.extract(
            pil_image
        )

        handcrafted.append(
            result.handcrafted
        )
        deep.append(
            result.deep
        )
        labels.append(
            int(label)
        )

    handcrafted = np.stack(
        handcrafted
    )
    deep = np.stack(deep)
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    return handcrafted, deep, labels


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract handcrafted + ResNet18 deep features "
            "from PCam patches."
        )
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
        default=str(
            DATA_DIR
            / "hybrid"
            / "features"
        ),
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    make_output_dirs()

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()
    print(f"Feature device: {device}")

    checkpoint = Path(
        args.checkpoint
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"ResNet checkpoint not found: {checkpoint}"
        )

    extractor = ImageFeatureExtractor(
        checkpoint_path=checkpoint,
        device=device,
    )

    metadata = {
        "source": "PCam",
        "device": str(device),
        "resnet_checkpoint": str(
            checkpoint.resolve()
        ),
        "handcrafted_feature_count": None,
        "deep_feature_count": 512,
        "splits": {},
    }

    for split, limit in (
        ("train", args.max_train),
        ("val", args.max_val),
        ("test", args.max_test),
    ):
        h, d, y = process_split(
            split,
            limit,
            extractor,
        )

        np.save(
            output / f"{split}_handcrafted.npy",
            h,
        )
        np.save(
            output / f"{split}_deep.npy",
            d,
        )
        np.save(
            output / f"{split}_labels.npy",
            y,
        )

        metadata["splits"][split] = {
            "samples": int(len(y)),
            "handcrafted_shape": list(h.shape),
            "deep_shape": list(d.shape),
        }

        metadata["handcrafted_feature_count"] = int(
            h.shape[1]
        )

    names = feature_names()

    (output / "handcrafted_feature_names.json").write_text(
        json.dumps(names, indent=2),
        encoding="utf-8",
    )

    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nHybrid feature extraction complete.")
    print(
        f"Handcrafted features: "
        f"{metadata['handcrafted_feature_count']}"
    )
    print(
        "Deep features       : 512"
    )
    print(
        f"Output directory    : {output}"
    )


if __name__ == "__main__":
    main()

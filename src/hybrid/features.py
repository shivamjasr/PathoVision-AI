from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.datasets.pcam import MEAN, STD
from src.models.resnet18 import build_resnet18


@dataclass(frozen=True)
class FeatureResult:
    handcrafted: np.ndarray
    deep: np.ndarray


def _safe_stats(values):
    values = values.astype(np.float32)

    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        float(values.mean()),
        float(values.std()),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    ]


def _glcm_features(gray_uint8, levels=16):
    """Small normalized GLCM-like texture descriptor.

    We quantize grayscale intensities and estimate horizontal and
    vertical co-occurrence matrices. Four standard texture statistics
    are returned for each direction.
    """
    if gray_uint8.ndim != 2:
        raise ValueError("gray_uint8 must be a 2D array.")

    quantized = (
        gray_uint8.astype(np.uint16) * levels // 256
    ).clip(0, levels - 1)

    matrices = []

    for dy, dx in ((0, 1), (1, 0)):
        source = quantized[
            0 : quantized.shape[0] - dy if dy else quantized.shape[0],
            0 : quantized.shape[1] - dx if dx else quantized.shape[1],
        ]

        target = quantized[
            dy: quantized.shape[0],
            dx: quantized.shape[1],
        ]

        matrix = np.zeros(
            (levels, levels),
            dtype=np.float64,
        )

        np.add.at(
            matrix,
            (source.ravel(), target.ravel()),
            1.0,
        )

        # Symmetrize so the descriptor does not depend strongly on
        # the chosen direction.
        matrix = matrix + matrix.T
        total = matrix.sum()

        if total > 0:
            matrix /= total

        matrices.append(matrix)

    values = []

    for matrix in matrices:
        i, j = np.indices(matrix.shape)

        contrast = np.sum(
            matrix * (i - j) ** 2
        )

        dissimilarity = np.sum(
            matrix * np.abs(i - j)
        )

        homogeneity = np.sum(
            matrix / (1.0 + np.abs(i - j))
        )

        energy = np.sum(matrix ** 2)

        values.extend(
            [
                float(contrast),
                float(dissimilarity),
                float(homogeneity),
                float(energy),
            ]
        )

    return values


def handcrafted_features(image: Image.Image) -> np.ndarray:
    """Extract interpretable color, texture and morphology features."""
    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV,
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY,
    )

    features = []

    # RGB statistics
    for channel in cv2.split(rgb):
        features.extend(_safe_stats(channel))

    # HSV statistics
    for channel in cv2.split(hsv):
        features.extend(_safe_stats(channel))

    # Texture
    features.extend(
        _glcm_features(gray)
    )

    # Basic edge density
    edges = cv2.Canny(
        gray,
        threshold1=50,
        threshold2=150,
    )
    features.append(
        float((edges > 0).mean())
    )

    # Threshold darker tissue-like structures and summarize morphology.
    _, binary = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    areas = []
    perimeters = []
    circularities = []
    aspect_ratios = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < 5:
            continue

        perimeter = float(
            cv2.arcLength(contour, True)
        )

        if perimeter <= 0:
            continue

        circularity = (
            4.0 * np.pi * area
            / (perimeter * perimeter)
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        aspect = (
            float(w) / float(h)
            if h > 0
            else 0.0
        )

        areas.append(area)
        perimeters.append(perimeter)
        circularities.append(circularity)
        aspect_ratios.append(aspect)

    for values in (
        areas,
        perimeters,
        circularities,
        aspect_ratios,
    ):
        features.extend(
            _safe_stats(
                np.asarray(
                    values,
                    dtype=np.float32,
                )
            )
        )

    # Global stain-like ratios.
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)

    denominator = (
        r + g + b + 1e-6
    )

    for channel in (r, g, b):
        ratio = channel / denominator
        features.extend(
            _safe_stats(ratio)
        )

    return np.asarray(
        features,
        dtype=np.float32,
    )


def load_resnet_feature_extractor(
    checkpoint_path,
    device,
):
    """Load ResNet18 and expose its 512-d feature vector."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = build_resnet18(
        pretrained=False,
        strategy="full",
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ResNet18's final classifier is replaced with Identity, so forward
    # returns the 512-dimensional penultimate representation.
    model.fc = torch.nn.Identity()

    return model


class ImageFeatureExtractor:
    def __init__(
        self,
        checkpoint_path,
        device=None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.device = device

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    MEAN,
                    STD,
                ),
            ]
        )

        self.model = load_resnet_feature_extractor(
            checkpoint_path,
            device,
        )

    def deep_feature(self, image: Image.Image):
        tensor = self.transform(
            image.convert("RGB")
        ).unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():
            feature = self.model(
                tensor
            )[0]

        return feature.cpu().numpy().astype(
            np.float32
        )

    def extract(
        self,
        image: Image.Image,
    ):
        classical = handcrafted_features(
            image
        )

        deep = self.deep_feature(
            image
        )

        return FeatureResult(
            handcrafted=classical,
            deep=deep,
        )


def feature_names():
    names = []

    for color_space, channels in (
        ("RGB", 3),
        ("HSV", 3),
    ):
        for channel in range(channels):
            for stat in (
                "mean",
                "std",
                "p25",
                "p75",
            ):
                names.append(
                    f"{color_space}_{channel}_{stat}"
                )

    for direction in (
        "horizontal",
        "vertical",
    ):
        for stat in (
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
        ):
            names.append(
                f"GLCM_{direction}_{stat}"
            )

    names.append("edge_density")

    for morphology in (
        "area",
        "perimeter",
        "circularity",
        "aspect_ratio",
    ):
        for stat in (
            "mean",
            "std",
            "p25",
            "p75",
        ):
            names.append(
                f"morph_{morphology}_{stat}"
            )

    for channel in (
        "R",
        "G",
        "B",
    ):
        for stat in (
            "mean",
            "std",
            "p25",
            "p75",
        ):
            names.append(
                f"ratio_{channel}_{stat}"
            )

    return names

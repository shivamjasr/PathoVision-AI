from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def build_probability_maps(x: Sequence[int], y: Sequence[int], width: Sequence[int], height: Sequence[int], probabilities: Sequence[float], slide_width: int, slide_height: int, thumbnail_size: Tuple[int, int]):
    """Project level-0 tile probabilities into thumbnail coordinates."""
    thumb_width, thumb_height = thumbnail_size
    if slide_width <= 0 or slide_height <= 0:
        raise ValueError("Slide dimensions must be positive.")

    prob_sum = np.zeros((thumb_height, thumb_width), dtype=np.float32)
    coverage = np.zeros((thumb_height, thumb_width), dtype=np.float32)

    for xi, yi, wi, hi, prob in zip(x, y, width, height, probabilities):
        tx0 = int(round(xi * thumb_width / slide_width))
        ty0 = int(round(yi * thumb_height / slide_height))
        tx1 = int(round((xi + wi) * thumb_width / slide_width))
        ty1 = int(round((yi + hi) * thumb_height / slide_height))

        tx0 = max(0, min(tx0, thumb_width)); tx1 = max(0, min(tx1, thumb_width))
        ty0 = max(0, min(ty0, thumb_height)); ty1 = max(0, min(ty1, thumb_height))
        if tx1 <= tx0 or ty1 <= ty0:
            continue

        prob_sum[ty0:ty1, tx0:tx1] += float(prob)
        coverage[ty0:ty1, tx0:tx1] += 1.0

    probability_map = np.full_like(prob_sum, np.nan)
    covered = coverage > 0
    probability_map[covered] = prob_sum[covered] / coverage[covered]
    return probability_map, coverage


def save_heatmap(probability_map: np.ndarray, output_path: str | Path):
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    filled = np.nan_to_num(probability_map, nan=0.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(filled, vmin=0, vmax=1)
    ax.set_title("Tumor probability heatmap")
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Tumor probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_overlay(thumbnail_path: str | Path, probability_map: np.ndarray, output_path: str | Path, alpha: float = 0.45):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail = Image.open(thumbnail_path).convert("RGB")
    if (thumbnail.width, thumbnail.height) != (probability_map.shape[1], probability_map.shape[0]):
        raise ValueError("Thumbnail and probability-map dimensions do not match.")

    valid = np.isfinite(probability_map)
    normalized = np.nan_to_num(probability_map, nan=0.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(thumbnail)
    heat = ax.imshow(normalized, vmin=0, vmax=1, alpha=np.where(valid, alpha, 0.0))
    ax.set_title("WSI tumor-probability overlay")
    ax.axis("off")
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04, label="Tumor probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path

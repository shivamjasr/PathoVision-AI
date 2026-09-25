from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PCAM_DIR = DATA_DIR / "pcam"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"

IMAGE_SIZE = 96
BATCH_SIZE = 128
NUM_WORKERS = 2

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

CLASS_NAMES = {
    0: "normal",
    1: "tumor",
}


def get_device() -> torch.device:
    """Pick CUDA, then MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def make_output_dirs() -> None:
    for path in (PCAM_DIR, CHECKPOINT_DIR, PLOTS_DIR, METRICS_DIR):
        path.mkdir(parents=True, exist_ok=True)

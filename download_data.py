import argparse

from src.config import PCAM_DIR, make_output_dirs
from src.datasets.pcam import get_pcam_dataset


def main():
    parser = argparse.ArgumentParser(description="Download PatchCamelyon.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    args = parser.parse_args()

    make_output_dirs()
    print(f"PCam root: {PCAM_DIR.resolve()}")

    for split in args.splits:
        print(f"\nPreparing split: {split}")
        dataset = get_pcam_dataset(split, download=True)
        print(f"{split}: {len(dataset):,} samples")

    print("\nPCam setup complete.")


if __name__ == "__main__":
    main()

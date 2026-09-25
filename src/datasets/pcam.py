from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import BATCH_SIZE, NUM_WORKERS, PCAM_DIR


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_transforms(train: bool, image_size: int = 96):
    """Transforms for the baseline CNN.

    PCam samples are 96x96. For ResNet18, use image_size=224.
    """
    operations = []

    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
            ]
        )

    operations.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    return transforms.Compose(operations)


def get_pcam_dataset(
    split: str,
    download: bool = False,
    transform=None,
    image_size: int = 96,
):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")

    if transform is None:
        transform = get_transforms(
            train=(split == "train"),
            image_size=image_size,
        )

    return datasets.PCAM(
        root=str(PCAM_DIR),
        split=split,
        transform=transform,
        download=download,
    )


def maybe_limit(dataset, max_samples: Optional[int]):
    if max_samples is None or max_samples <= 0:
        return dataset

    max_samples = min(max_samples, len(dataset))
    return Subset(dataset, range(max_samples))


def make_loaders(
    download: bool = False,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    limit_test: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    image_size: int = 96,
):
    train_ds = get_pcam_dataset(
        "train",
        download=download,
        image_size=image_size,
    )
    val_ds = get_pcam_dataset(
        "val",
        download=download,
        image_size=image_size,
    )
    test_ds = get_pcam_dataset(
        "test",
        download=download,
        image_size=image_size,
    )

    train_ds = maybe_limit(train_ds, limit_train)
    val_ds = maybe_limit(val_ds, limit_val)
    test_ds = maybe_limit(test_ds, limit_test)

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader

"""
dataset.py — CIFAR-10 loader with stratified splits
Person 1 deliverable for ML & AI 30562, Bocconi University

Usage (CLI):
    python dataset.py --fraction 0.1 --seed 42

Usage (as module):
    from data.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(fraction=0.1, seed=42)
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# CIFAR-10 channel mean and std (computed over full training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# Fraction of the training set carved out as validation
VAL_FRACTION = 0.10

VALID_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def get_transforms(augment=False):
    """
    Returns torchvision transform pipelines.

    Args:
        augment (bool): If True, applies RandAugment on top of the baseline
                        transforms (used for the augmentation extension in step 5).
                        Baseline runs must pass augment=False.
    """
    base = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    if augment:
        # RandAugment settings agreed by the group: magnitude=9, num_ops=2
        aug = [transforms.RandAugment(num_ops=2, magnitude=9)]
        train_transform = transforms.Compose(aug + base)
    else:
        train_transform = transforms.Compose(base)

    test_transform = transforms.Compose(base)  # never augment val/test
    return train_transform, test_transform


def get_dataloaders(
    fraction: float,
    seed: int,
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = False,
):
    """
    Build train / val / test DataLoaders for CIFAR-10.

    Args:
        fraction   : Fraction of the 50k training set to use.
                     Must be one of: 0.01, 0.05, 0.10, 0.25, 0.50, 1.00
        seed       : Random seed for reproducibility.
        data_root  : Directory where CIFAR-10 is downloaded / cached.
        batch_size : Mini-batch size for all loaders.
        num_workers: DataLoader worker processes.
        augment    : If True, applies RandAugment to the training loader.

    Returns:
        train_loader, val_loader, test_loader
    """
    if fraction not in VALID_FRACTIONS:
        raise ValueError(
            f"fraction must be one of {VALID_FRACTIONS}, got {fraction}"
        )

    # Seed Python/NumPy/PyTorch for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_transform, test_transform = get_transforms(augment=augment)

    # Download CIFAR-10 if not already present
    full_train = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    labels = np.array(full_train.targets)
    all_indices = np.arange(len(labels))  # 0 … 49 999

    # ------------------------------------------------------------------ #
    # Step 1: carve out a stratified validation set (VAL_FRACTION of 50k) #
    # ------------------------------------------------------------------ #
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=VAL_FRACTION,
        stratify=labels[all_indices],
        random_state=seed,
    )

    # ------------------------------------------------------------------ #
    # Step 2: subsample the remaining training indices by `fraction`       #
    # ------------------------------------------------------------------ #
    if fraction < 1.0:
        train_idx, _ = train_test_split(
            train_idx,
            train_size=fraction,
            stratify=labels[train_idx],
            random_state=seed,
        )

    train_subset = Subset(full_train, train_idx)
    val_subset   = Subset(full_train, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"[dataset] fraction={fraction} | seed={seed} | augment={augment}")
    print(f"[dataset] train samples : {len(train_subset)}")
    print(f"[dataset] val   samples : {len(val_subset)}")
    print(f"[dataset] test  samples : {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------------ #
# CLI entry point — quick sanity check                                #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 dataloader sanity check")
    parser.add_argument("--fraction", type=float, default=0.1,
                        help="Training fraction (0.01/0.05/0.10/0.25/0.50/1.00)")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--data_root",type=str,   default="./data")
    parser.add_argument("--batch_size",type=int,  default=128)
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(
        fraction=args.fraction,
        seed=args.seed,
        data_root=args.data_root,
        batch_size=args.batch_size,
    )

    # Check one batch shape
    images, labels = next(iter(train_loader))
    print(f"[dataset] batch shape: {images.shape}, labels shape: {labels.shape}")
    print("[dataset] All checks passed.")

"""
train.py — Training loop for CNN vs ViT under data scarcity
Person 1 deliverable for ML & AI 30562, Bocconi University

Usage (CLI):
    python train.py --model cnn --fraction 0.1 --seed 42 --epochs 100 --lr 1e-3 --weight_decay 1e-4

Usage (as module, e.g. from Person 4's run queue):
    from train import run
    run(model_name="cnn", fraction=0.1, seed=42, epochs=100, lr=1e-3, weight_decay=1e-4)
"""

import argparse
import csv
import importlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Make sure repo root is on the path (important when imported as a module)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import get_dataloaders

# ------------------------------------------------------------------ #
# Paths                                                                #
# ------------------------------------------------------------------ #
RESULTS_DIR     = "results"
EXPERIMENTS_DIR = "experiments"
MASTER_CSV      = os.path.join(RESULTS_DIR, "master.csv")
AUGMENT_CSV     = os.path.join(RESULTS_DIR, "augmentation.csv")

CSV_COLUMNS = [
    "model", "fraction", "seed", "epoch",
    "train_loss", "train_acc", "val_loss", "val_acc", "test_acc"
]


# ------------------------------------------------------------------ #
# Helper: load model dynamically from /models/{name}.py               #
# ------------------------------------------------------------------ #
def get_model(model_name: str) -> nn.Module:
    """
    Dynamically imports /models/{model_name}.py and calls its get_model()
    function. This means no architecture is hardcoded here.
    """
    module_path = f"models.{model_name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        raise FileNotFoundError(
            f"Could not find models/{model_name}.py — "
            f"make sure Person 2/3 have committed their model file."
        )
    if not hasattr(module, "get_model"):
        raise AttributeError(
            f"models/{model_name}.py must define a get_model() function."
        )
    return module.get_model()


# ------------------------------------------------------------------ #
# Helper: CSV logging                                                  #
# ------------------------------------------------------------------ #
def append_csv_row(csv_path: str, row: dict):
    """Appends one row to a CSV. Creates file with header if it doesn't exist."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ------------------------------------------------------------------ #
# One epoch of training                                                #
# ------------------------------------------------------------------ #
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ------------------------------------------------------------------ #
# Evaluation (val or test)                                             #
# ------------------------------------------------------------------ #
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ------------------------------------------------------------------ #
# Parameter count check                                                #
# ------------------------------------------------------------------ #
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ #
# Main training function                                               #
# ------------------------------------------------------------------ #
def run(
    model_name:   str,
    fraction:     float,
    seed:         int,
    epochs:       int   = 100,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size:   int   = 128,
    data_root:    str   = "./data",
    augment:      bool  = False,
    num_workers:  int   = 2,
):
    """
    Full training run for one (model, fraction, seed) combination.

    Args:
        model_name  : Must match a file in /models/ (e.g. 'cnn' or 'vit')
        fraction    : Data fraction — one of 0.01/0.05/0.10/0.25/0.50/1.00
        seed        : Random seed for full reproducibility
        epochs      : Number of training epochs
        lr          : Initial learning rate for AdamW
        weight_decay: Weight decay for AdamW
        batch_size  : Mini-batch size
        data_root   : Path where CIFAR-10 is downloaded/cached
        augment     : If True, applies RandAugment (for step 5)
                      Results logged to augmentation.csv instead of master.csv
        num_workers : DataLoader workers
    """
    # Seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  model={model_name} | fraction={fraction} | seed={seed} | augment={augment}")
    print(f"  epochs={epochs} | lr={lr} | wd={weight_decay} | device={device}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        fraction=fraction,
        seed=seed,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment,
    )

    # Model
    model = get_model(model_name).to(device)
    n_params = count_parameters(model)
    print(f"[train] {model_name} parameters: {n_params:,}")

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Checkpoint directory
    run_id   = f"{model_name}_{fraction}_{seed}"
    ckpt_dir = os.path.join(EXPERIMENTS_DIR, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, "best.pth")

    # CSV target
    csv_path = AUGMENT_CSV if augment else MASTER_CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_val_acc = 0.0
    test_acc     = 0.0   # computed once at the end using best checkpoint

    # Training loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state": model.state_dict(),
                    "val_acc":    val_acc,
                    "optimizer":  optimizer.state_dict(),
                },
                best_ckpt,
            )

        # Log epoch row (test_acc left as 0.0 until final eval below)
        append_csv_row(csv_path, {
            "model":      model_name,
            "fraction":   fraction,
            "seed":       seed,
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "test_acc":   "",   # filled after final eval
        })

    # Final evaluation on test set using best checkpoint
    print(f"\n[train] Loading best checkpoint (val_acc={best_val_acc:.4f}) for test eval…")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[train] Final test accuracy: {test_acc:.4f}")

    # Append a summary row with test_acc filled in
    append_csv_row(csv_path, {
        "model":      model_name,
        "fraction":   fraction,
        "seed":       seed,
        "epoch":      "FINAL",
        "train_loss": "",
        "train_acc":  "",
        "val_loss":   "",
        "val_acc":    round(best_val_acc, 6),
        "test_acc":   round(test_acc, 6),
    })

    print(f"[train] Results appended to {csv_path}")
    print(f"[train] Best checkpoint saved to {best_ckpt}\n")

    return test_acc


# ------------------------------------------------------------------ #
# CLI entry point                                                       #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN or ViT on CIFAR-10")
    parser.add_argument("--model",        type=str,   required=True,
                        help="Model name matching a file in /models/ (e.g. cnn, vit)")
    parser.add_argument("--fraction",     type=float, required=True,
                        help="Training data fraction: 0.01/0.05/0.10/0.25/0.50/1.00")
    parser.add_argument("--seed",         type=int,   required=True,
                        help="Random seed")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--data_root",    type=str,   default="./data")
    parser.add_argument("--augment",      action="store_true",
                        help="Enable RandAugment (for augmentation extension, step 5)")
    parser.add_argument("--num_workers",  type=int,   default=2)
    args = parser.parse_args()

    run(
        model_name=args.model,
        fraction=args.fraction,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        data_root=args.data_root,
        augment=args.augment,
        num_workers=args.num_workers,
    )

"""
training/train_gat.py

Full training loop for the DialectGAT model.

Includes:
  - Weighted CrossEntropyLoss for class imbalance (Gulf Arabic is smallest)
  - AdamW optimizer
  - ReduceLROnPlateau scheduler (monitors val macro F1)
  - Early stopping with configurable patience
  - TensorBoard logging
  - Per-epoch Gulf Arabic F1 logging (hardest dialect, tracked separately)
  - Checkpoint saving on best val macro F1

Usage:
    python training/train_gat.py \
        --config configs/gat_base.yaml \
        --checkpoint_dir checkpoints/gat/ \
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports that depend on the project packages
# ---------------------------------------------------------------------------

try:
    from torch_geometric.loader import DataLoader as PygDataLoader  # type: ignore
except ImportError:
    from torch_geometric.data import DataLoader as PygDataLoader  # type: ignore

from data.dataset import ArabicDialectGraphDataset, get_dialect_weights, DIALECT_NAMES
from models.gat_model import DialectGAT, build_gat

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    logger.warning("TensorBoard not found; training metrics will only print to stdout.")

DIALECT_NAMES_LIST: List[str] = DIALECT_NAMES
GULF_IDX = DIALECT_NAMES_LIST.index("Gulf")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Per-epoch functions
# ---------------------------------------------------------------------------

def train_epoch(
    model: DialectGAT,
    loader: PygDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[object] = None,
) -> Dict[str, float]:
    """Run one training epoch.

    Args:
        model: DialectGAT model.
        loader: PyG DataLoader for training graphs.
        optimizer: AdamW optimizer.
        criterion: Weighted CrossEntropyLoss.
        device: Training device.
        scheduler: If a step-level scheduler (e.g. OneCycleLR), step here.
            For plateau-based schedulers, step outside this function.

    Returns:
        Dict with ``"loss"`` and ``"accuracy"`` over the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        # Gradient clipping prevents the occasional large gradient spike
        # from disrupting the attention weight learning.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=-1)
        correct += int((preds == batch.y).sum())
        total += batch.num_graphs

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def validate_epoch(
    model: DialectGAT,
    loader: PygDataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict:
    """Evaluate on validation set, returning macro F1 and per-dialect F1.

    Args:
        model: DialectGAT model (put in eval mode inside this function).
        loader: PyG DataLoader for validation graphs.
        device: Evaluation device.
        criterion: Loss function (for tracking val loss alongside F1).

    Returns:
        Dict with ``"macro_f1"``, ``"per_dialect_f1"`` (dict), ``"loss"``.
    """
    from sklearn.metrics import f1_score  # type: ignore

    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            loss = criterion(logits, batch.y)
            total_loss += loss.item() * batch.num_graphs
            total += batch.num_graphs

            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().tolist())

    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    per_class = f1_score(
        all_labels, all_preds, average=None, zero_division=0,
        labels=list(range(len(DIALECT_NAMES_LIST))),
    )
    per_dialect_f1 = {
        DIALECT_NAMES_LIST[i]: float(per_class[i]) for i in range(len(DIALECT_NAMES_LIST))
    }

    return {
        "macro_f1": macro_f1,
        "per_dialect_f1": per_dialect_f1,
        "loss": total_loss / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when val macro F1 stops improving.

    Args:
        patience: Epochs to wait without improvement before stopping.
        min_delta: Minimum improvement to count as an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best = -float("inf")
        self._counter = 0

    def step(self, metric: float) -> bool:
        """Call after each epoch with the monitored metric.

        Returns:
            True if training should stop, False otherwise.
        """
        if metric > self._best + self.min_delta:
            self._best = metric
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience

    @property
    def best(self) -> float:
        return self._best


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main(
    config_path: str,
    checkpoint_dir: str,
    resume: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """Full training pipeline for DialectGAT.

    Args:
        config_path: Path to YAML config file (e.g. configs/gat_base.yaml).
        checkpoint_dir: Directory to save best checkpoint.
        resume: Optional path to a checkpoint to resume from.
        seed: Random seed.

    Returns:
        Dict with final test results and training history.
    """
    set_seed(seed)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    opt_cfg = train_cfg.get("optimizer", {})
    sched_cfg = train_cfg.get("scheduler", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    data_root = data_cfg.get("data_dir", "data/graphs")
    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 0)   # 0 is safe on Windows

    train_ds = ArabicDialectGraphDataset(root=data_root, split="train")
    val_ds = ArabicDialectGraphDataset(root=data_root, split="val")
    test_ds = ArabicDialectGraphDataset(root=data_root, split="test")

    train_loader = PygDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = PygDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = PygDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info("Train: %d | Val: %d | Test: %d", len(train_ds), len(val_ds), len(test_ds))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_gat(config).to(device)
    logger.info("Model: %s", model)

    # ------------------------------------------------------------------
    # Loss: weighted CrossEntropy to compensate for Gulf class imbalance.
    # ------------------------------------------------------------------
    if train_cfg.get("weighted_loss", True):
        class_weights = get_dialect_weights(train_ds).to(device)
        logger.info("Using weighted loss. Weights: %s", class_weights.tolist())
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 1e-3),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",                            # We maximize macro F1
        patience=sched_cfg.get("patience", 5),
        factor=sched_cfg.get("factor", 0.5),
        min_lr=sched_cfg.get("min_lr", 1e-6),
    )
    early_stopping = EarlyStopping(
        patience=train_cfg.get("early_stopping_patience", 10)
    )

    # ------------------------------------------------------------------
    # Optionally resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 1
    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if _TB_AVAILABLE:
        tb_dir = ckpt_dir / "tb_logs"
        writer = SummaryWriter(log_dir=str(tb_dir))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    epochs = train_cfg.get("epochs", 100)
    best_ckpt_path = ckpt_dir / "gat_best.pth"
    history: List[Dict] = []

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_epoch(model, val_loader, device, criterion)

        macro_f1 = val_metrics["macro_f1"]
        gulf_f1 = val_metrics["per_dialect_f1"].get("Gulf", 0.0)

        scheduler.step(macro_f1)
        stop = early_stopping.step(macro_f1)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc":  train_metrics["accuracy"],
            "val_loss":   val_metrics["loss"],
            "val_macro_f1": macro_f1,
            "val_gulf_f1":  gulf_f1,
            **{f"val_{k}_f1": v for k, v in val_metrics["per_dialect_f1"].items()},
        }
        history.append(epoch_log)

        logger.info(
            "Ep %3d/%d  train_loss=%.4f  val_macro_f1=%.4f  gulf_f1=%.4f  lr=%.2e",
            epoch, epochs,
            train_metrics["loss"],
            macro_f1,
            gulf_f1,
            optimizer.param_groups[0]["lr"],
        )

        if writer:
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("F1/val_macro", macro_f1, epoch)
            writer.add_scalar("F1/val_gulf", gulf_f1, epoch)
            for dialect, f1 in val_metrics["per_dialect_f1"].items():
                writer.add_scalar(f"F1/{dialect}", f1, epoch)

        # Save best checkpoint.
        if macro_f1 >= early_stopping.best:
            model.save_checkpoint(
                str(best_ckpt_path),
                extra={
                    "epoch": epoch,
                    "val_macro_f1": macro_f1,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
            )
            logger.info("  ✓ New best checkpoint saved (macro F1 = %.4f)", macro_f1)

        if stop:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d).",
                epoch, early_stopping.patience,
            )
            break

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    logger.info("\n--- Final test evaluation ---")
    best_model = DialectGAT.from_pretrained(str(best_ckpt_path))
    best_model = best_model.to(device)
    test_metrics = validate_epoch(best_model, test_loader, device, criterion)

    logger.info("Test macro F1  : %.4f", test_metrics["macro_f1"])
    for dialect, f1 in test_metrics["per_dialect_f1"].items():
        logger.info("  %-12s : %.4f", dialect, f1)

    if writer:
        writer.close()

    return {
        "test_results": test_metrics,
        "history": history,
        "best_checkpoint": str(best_ckpt_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the DialectGAT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/gat_base.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/gat/")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    cli_main()

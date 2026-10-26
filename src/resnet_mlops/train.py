"""Training loop: train + validate, track with MLflow, checkpoint the best model.

The validation split drives model selection (best checkpoint + early stopping);
the test set is deliberately never used here — see evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from resnet_mlops.config import Config, load_config, resolve_device
from resnet_mlops.data import build_dataloaders
from resnet_mlops.model import ResNet

try:
    import mlflow
except ImportError:  # tracking stays optional
    mlflow = None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    t = cfg.train
    if t.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t.lr, momentum=t.momentum, weight_decay=t.weight_decay)
    if t.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    raise ValueError(f"Unknown optimizer: {t.optimizer}")


def build_scheduler(opt: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int) -> torch.optim.lr_scheduler.LRScheduler | None:
    t = cfg.train
    if t.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t.epochs * steps_per_epoch)
    if t.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    return None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    return run_epoch(model, loader, criterion, device, optimizer=None)


def train(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    bundle = build_dataloaders(
        dataset=cfg.data.dataset,
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
        seed=cfg.seed,
    )
    cfg.model.num_classes = bundle.num_classes
    model = ResNet(num_classes=bundle.num_classes, blocks=cfg.model.blocks).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(bundle.train))

    out_dir = Path(cfg.output.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / cfg.output.best_model_name
    last_path = out_dir / cfg.output.last_model_name

    ckpt_ctx = nullcontext()
    active_run = None
    if cfg.tracking.backend == "mlflow":
        if mlflow is None:
            raise RuntimeError("mlflow is not installed; set tracking.backend: none or pip install mlflow")
        if cfg.tracking.tracking_uri:
            mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
        mlflow.set_experiment(cfg.tracking.experiment_name)
        active_run = mlflow.start_run(run_name=cfg.tracking.run_name)
        ckpt_ctx = active_run
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "device": device,
                "dataset": cfg.data.dataset,
                "batch_size": cfg.data.batch_size,
                "val_split": cfg.data.val_split,
                "blocks": cfg.model.blocks,
                "num_classes": bundle.num_classes,
                "epochs": cfg.train.epochs,
                "lr": cfg.train.lr,
                "optimizer": cfg.train.optimizer,
                "scheduler": cfg.train.scheduler,
                "label_smoothing": cfg.train.label_smoothing,
                "model_class": model.__class__.__name__,
            }
        )
        mlflow.log_dict({"classes": bundle.class_names}, "classes.json")

    best_val_acc, best_epoch, patience_left = 0.0, -1, cfg.train.early_stopping_patience
    history: list[dict] = []
    start = time.time()
    with ckpt_ctx:
        for epoch in range(cfg.train.epochs):
            train_loss, train_acc = run_epoch(model, bundle.train, criterion, device, optimizer)
            val_loss, val_acc = validate(model, bundle.val, criterion, device)
            if scheduler is not None:
                scheduler.step()

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc, best_epoch = val_acc, epoch
                patience_left = cfg.train.early_stopping_patience
                torch.save(_checkpoint(model, cfg, bundle.class_names, epoch, best_val_acc), best_path)
            else:
                patience_left -= 1

            torch.save(_checkpoint(model, cfg, bundle.class_names, epoch, best_val_acc), last_path)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(
                f"epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                f"best {best_val_acc:.4f}@{best_epoch}{' *' if improved else ''}"
            )
            if cfg.tracking.backend == "mlflow":
                mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch} (best {best_val_acc:.4f} @ {best_epoch})")
                break

        if cfg.tracking.backend == "mlflow":
            mlflow.log_metric("best_val_acc", best_val_acc)
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_metric("train_seconds", time.time() - start)
            mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
            mlflow.log_dict({"history": history}, "history.json")
            run_id = mlflow.active_run().info.run_id
            mlflow.end_run()
        else:
            run_id = None

    summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "run_id": run_id,
        "history": history,
        "device": device,
        "classes": bundle.class_names,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _checkpoint(model: nn.Module, cfg: Config, class_names: list[str], epoch: int, val_acc: float) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": {"num_classes": cfg.model.num_classes, "blocks": list(cfg.model.blocks)},
        "classes": class_names,
        "epoch": epoch,
        "val_acc": val_acc,
    }


def _parse_overrides(pairs: list[str]) -> dict:
    overrides: dict = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        parts = key.split(".")
        node = overrides
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = json.loads(value)
        except json.JSONDecodeError:
            node[parts[-1]] = value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5 data.batch_size=64")
    args = parser.parse_args()
    summary = train(load_config(args.config, _parse_overrides(args.set)))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()

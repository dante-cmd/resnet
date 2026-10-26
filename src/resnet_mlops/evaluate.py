"""Test-set evaluation: loads a checkpoint and reports standard classification
metrics. The test set is used exactly once per model selection, never for
hyperparameter tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from resnet_mlops.config import load_config, resolve_device
from resnet_mlops.data import build_dataloaders
from resnet_mlops.model import ResNet

try:
    import mlflow
except ImportError:
    mlflow = None

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def load_checkpoint(path: str | Path, device: str) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ResNet(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate(cfg, checkpoint: str) -> dict:
    device = resolve_device(cfg.device)
    model, ckpt = load_checkpoint(checkpoint, device)
    bundle = build_dataloaders(
        dataset=cfg.data.dataset,
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
        seed=cfg.seed,
    )
    criterion = nn.CrossEntropyLoss()

    all_preds, all_targets, total_loss = [], [], 0.0
    for x, y in bundle.test:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_targets.extend(y.cpu().tolist())

    classes = ckpt.get("classes") or bundle.class_names
    metrics = {
        "checkpoint": str(checkpoint),
        "test_loss": total_loss / len(bundle.test.dataset),
        "test_accuracy": accuracy_score(all_targets, all_preds),
        "test_precision_macro": precision_score(all_targets, all_preds, average="macro", zero_division=0),
        "test_recall_macro": recall_score(all_targets, all_preds, average="macro", zero_division=0),
        "test_f1_macro": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(all_targets, all_preds).tolist(),
        "classification_report": classification_report(all_targets, all_preds, target_names=classes, zero_division=0),
    }

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    (out / "confusion_matrix.json").write_text(json.dumps(metrics["confusion_matrix"]))

    if cfg.tracking.backend == "mlflow" and mlflow is not None and ckpt.get("run_id"):
        with mlflow.start_run(run_id=ckpt["run_id"]):
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.log_artifact(str(out / "test_metrics.json"))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test set")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    metrics = evaluate(load_config(args.config), args.checkpoint)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()

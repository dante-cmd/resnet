"""Typed configuration loaded from YAML, with CLI overrides.

Usage:
    cfg = load_config("configs/train.yaml")
    cfg = load_config("configs/train.yaml", overrides={"train": {"epochs": 5}})
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    root: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    val_split: float = 0.1


@dataclass
class ModelConfig:
    num_classes: int = 10
    blocks: list[int] = field(default_factory=lambda: [2, 2, 2, 2])


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5.0e-4
    optimizer: str = "sgd"
    scheduler: str = "cosine"
    label_smoothing: float = 0.1
    early_stopping_patience: int = 5


@dataclass
class TrackingConfig:
    backend: str = "mlflow"
    experiment_name: str = "resnet"
    run_name: str | None = None
    tracking_uri: str | None = None


@dataclass
class OutputConfig:
    checkpoint_dir: str = "checkpoints"
    best_model_name: str = "best.pt"
    last_model_name: str = "last.pt"


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _apply_overrides(obj: Any, overrides: dict[str, Any]) -> None:
    fields = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in overrides.items():
        if key not in fields:
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path and Path(path).exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    cfg = Config()
    _apply_overrides(cfg, raw)
    if overrides:
        _apply_overrides(cfg, overrides)
    return cfg


def resolve_device(preference: str) -> str:
    if preference != "auto":
        return preference
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

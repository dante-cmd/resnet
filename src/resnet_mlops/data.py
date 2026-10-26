"""Data pipelines: CIFAR-10/100 or a folder dataset, with a train/val/test split.

Returns canonical splits so train/evaluate agree on what "validate" and "test"
mean: the val split is used for model selection during training; the test set
is only touched by `evaluate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    num_classes: int
    class_names: list[str]


def _train_transform(mean: tuple, std: tuple) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _eval_transform(mean: tuple, std: tuple) -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def _class_subset(dataset: Dataset, frac: float, seed: int) -> Subset:
    """Hold out `frac` of the dataset (stratified) as a validation split."""
    if frac <= 0:
        raise ValueError("val_split must be > 0")
    g = torch.Generator().manual_seed(seed)
    n_total = len(dataset)
    n_val = int(n_total * frac)
    perm = torch.randperm(n_total, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    # If the dataset exposes labels, stratify: keep class balance in the val split.
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        by_class: dict[int, list[int]] = {}
        for idx in perm:
            by_class.setdefault(int(targets[idx]), []).append(idx)
        val_idx, train_idx = [], []
        for idxs in by_class.values():
            k = max(1, int(len(idxs) * frac))
            val_idx.extend(idxs[:k])
            train_idx.extend(idxs[k:])
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def build_dataloaders(
    dataset: str,
    root: str,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
) -> DataBundle:
    dataset = dataset.lower()
    if dataset in {"cifar10", "cifar100"}:
        mean, std = CIFAR_MEAN, CIFAR_STD
        cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
        train_full = cls(root, train=True, download=True, transform=_train_transform(mean, std))
        test_ds = cls(root, train=False, download=True, transform=_eval_transform(mean, std))
        num_classes = 10 if dataset == "cifar10" else 100
        class_names = list(train_full.classes)
    elif dataset == "imagefolder":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        train_full = datasets.ImageFolder(f"{root}/train", transform=_train_transform(mean, std))
        test_ds = datasets.ImageFolder(f"{root}/test", transform=_eval_transform(mean, std))
        num_classes = len(train_full.classes)
        class_names = list(train_full.classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_ds, val_ds = _class_subset(train_full, val_split, seed)

    def loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return DataBundle(
        train=loader(train_ds, True),
        val=loader(val_ds, False),
        test=loader(test_ds, False),
        num_classes=num_classes,
        class_names=class_names,
    )

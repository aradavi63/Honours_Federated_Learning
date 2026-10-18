"""Shared training configuration, datasets, models and protocol output helpers."""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch
from torchvision import datasets, transforms

from models.networks import Cifar10cnn, Mnistcnn
from training.data_paths import dataset_directory
from training.partitions import sample_dirichlet_train_data

DATASET_DETAILS = {
    "MNIST": {
        "model": "Mnistcnn",
        "mean": (0.1307,),
        "std": (0.3081,),
        "shape": (1, 28, 28),
    },
    "CIFAR10": {
        "model": "Cifar10cnn",
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "shape": (3, 32, 32),
    },
}


@dataclass(frozen=True)
class TrainingConfig:
    trial_seed: int = 4010
    dataset: str = "MNIST"
    rounds: int = 20
    num_users: int = 10
    samples_per_client: int = 12
    alpha: float = 10.0
    local_epochs: int = 5
    local_batch_size: int = 4
    local_lr: float = 0.01
    device: str = "cpu"
    momentum: float = 0.5
    eval_batch_size: int = 128
    data_root: str = None
    iid: bool = False
    partition_size: int = None
    exclusion_file: str = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _clone_state(state: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, value.detach().cpu().clone()) for name, value in state.items()
    )


class DummyCanvas:
    def draw(self):
        return None


class DummyAxis:
    def clear(self):
        return None


def _load_dataset(config: TrainingConfig):
    if config.dataset == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        root = dataset_directory("mnist", config.data_root)
        return (
            datasets.MNIST(root, train=True, download=True, transform=transform),
            datasets.MNIST(root, train=False, download=True, transform=transform),
        )
    if config.dataset == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )
        root = dataset_directory("cifar10", config.data_root)
        return (
            datasets.CIFAR10(root, train=True, download=True, transform=transform),
            datasets.CIFAR10(root, train=False, download=True, transform=transform),
        )
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def _sample_partitions(dataset, config: TrainingConfig, args):
    from torch.utils.data import Subset

    from training.exclusions import load_exclusions

    excluded = load_exclusions(config.exclusion_file, config.dataset, len(dataset))
    allowed = [i for i in range(len(dataset)) if i not in excluded]
    if len(allowed) < config.num_users * config.samples_per_client:
        raise ValueError("Insufficient training records after exclusions")
    sampling_data = Subset(dataset, allowed) if excluded else dataset
    _, sampled = sample_dirichlet_train_data(
        sampling_data, config.num_users, config.samples_per_client, args
    )
    partitions = OrderedDict(
        (client + 1, [allowed[int(index)] for index in sampled[client]])
        for client in range(config.num_users)
    )
    if any(len(values) != config.samples_per_client for values in partitions.values()):
        raise RuntimeError("A sampled client partition has the wrong size.")
    return partitions


def _build_model(config: TrainingConfig):
    args = SimpleNamespace(num_classes=10)
    if config.dataset == "MNIST":
        return Mnistcnn(args).to(config.device)
    return Cifar10cnn(args).to(config.device)


def _build_node_args(config: TrainingConfig, output_directory: Path):
    if config.dataset not in DATASET_DETAILS:
        raise ValueError("Unsupported experiment dataset")
    if (
        config.num_users < 3
        or min(
            config.rounds,
            config.samples_per_client,
            config.local_epochs,
            config.local_batch_size,
        )
        < 1
    ):
        raise ValueError("Require at least three clients and positive training counts")
    if (
        not np.isfinite([config.alpha, config.local_lr, config.momentum]).all()
        or config.alpha <= 0
    ):
        raise ValueError("Training parameters must be finite and alpha positive")
    if (
        config.partition_size is not None
        and not 1 <= config.partition_size <= config.num_users
    ):
        raise ValueError("Invalid SIHEFL route size")
    if (
        not 0 <= config.momentum < 1
        or config.eval_batch_size < 1
        or config.local_lr <= 0
    ):
        raise ValueError(
            "Require 0 <= momentum < 1, positive evaluation batch size and learning rate"
        )
    return SimpleNamespace(
        trial_seed=config.trial_seed,
        epochs=config.rounds,
        num_users=config.num_users,
        num_samples=config.samples_per_client,
        alpha=config.alpha,
        frac=1.0,
        local_ep=config.local_epochs,
        local_bs=config.local_batch_size,
        bs=config.eval_batch_size,
        lr=config.local_lr,
        momentum=config.momentum,
        model="cnn",
        partition_size=config.partition_size or config.num_users,
        output_directory=str(output_directory),
        dataset=config.dataset,
        data_root=config.data_root,
        iid=config.iid,
        num_classes=10,
        gpu=-1,
        all_clients=True,
        device=torch.device(config.device),
    )

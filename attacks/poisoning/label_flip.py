"""MNIST and CIFAR-10 label mappings and client-side label flips."""

import random
from dataclasses import dataclass
from typing import Callable, FrozenSet, Iterable, Mapping, Optional

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ClassMapping:
    name: str
    source_class: int
    target_class: int
    source_name: str
    target_name: str
    replace: Callable


def replace_5_with_3(label):
    return _replace(label, 5, 3)


def replace_0_with_2(label):
    return _replace(label, 0, 2)


def replace_1_with_9(label):
    return _replace(label, 1, 9)


CIFAR10_MAPPINGS: Mapping[str, ClassMapping] = {
    "dog_to_cat": ClassMapping("dog_to_cat", 5, 3, "dog", "cat", replace_5_with_3),
    "airplane_to_bird": ClassMapping(
        "airplane_to_bird", 0, 2, "airplane", "bird", replace_0_with_2
    ),
    "automobile_to_truck": ClassMapping(
        "automobile_to_truck", 1, 9, "automobile", "truck", replace_1_with_9
    ),
}


def replace_4_with_9(label):
    return _replace(label, 4, 9)


def replace_0_with_8(label):
    return _replace(label, 0, 8)


def replace_0_with_1(label):
    return _replace(label, 0, 1)


MNIST_MAPPINGS = {
    "4_to_9": ClassMapping("4_to_9", 4, 9, "4", "9", replace_4_with_9),
    "0_to_8": ClassMapping("0_to_8", 0, 8, "0", "8", replace_0_with_8),
    "0_to_1": ClassMapping("0_to_1", 0, 1, "0", "1", replace_0_with_1),
}
DATASET_MAPPINGS = {"MNIST": MNIST_MAPPINGS, "CIFAR10": CIFAR10_MAPPINGS}


@dataclass(frozen=True)
class AttackConfig:
    enabled: bool
    mapping: ClassMapping
    malicious_client_ids: FrozenSet[int]
    poison_fraction: float = 1.0
    start_round: int = 1
    end_round: Optional[int] = None
    seed: int = 4010

    def __post_init__(self):
        if not 0 <= self.poison_fraction <= 1:
            raise ValueError("poison-fraction must be between 0 and 1")
        if self.start_round < 1 or (
            self.end_round is not None and self.end_round < self.start_round
        ):
            raise ValueError("Invalid poisoning round interval")

    def active_at(self, round_number):
        return (
            self.enabled
            and round_number >= self.start_round
            and (self.end_round is None or round_number <= self.end_round)
        )


def choose_poisoned_indices(dataset, indices, source_class, fraction, seed):
    if not 0 <= fraction <= 1:
        raise ValueError("poison-fraction must be between 0 and 1")
    targets = getattr(dataset, "targets", None)
    eligible = [
        int(i)
        for i in indices
        if int(targets[i] if targets is not None else dataset[i][1]) == source_class
    ]
    return frozenset(
        random.Random(seed).sample(eligible, int(len(eligible) * fraction))
    )


class LabelFlipDataset(Dataset):
    """Read a dataset with one class label replaced."""

    def __init__(
        self, dataset: Dataset, mapping: ClassMapping, indices=None, on_flip=None
    ):
        self.dataset = dataset
        self.mapping = mapping
        self.indices = None if indices is None else frozenset(indices)
        self.on_flip = on_flip

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        feature, label = self.dataset[index]
        changed = (
            self.mapping.replace(label)
            if self.indices is None or index in self.indices
            else label
        )
        if self.on_flip is not None and _label_as_int(changed) != _label_as_int(label):
            self.on_flip(index, _label_as_int(label), _label_as_int(changed))
        return feature, changed


def select_malicious_clients(
    num_users: int, malicious_fraction: float, trial_seed: int
) -> FrozenSet[int]:
    if num_users < 1:
        raise ValueError("num_users must be positive")
    if not 0.0 <= malicious_fraction <= 1.0:
        raise ValueError("malicious_fraction must be between 0 and 1")

    malicious_count = int(round(num_users * malicious_fraction))
    if malicious_fraction > 0.0 and malicious_count == 0:
        raise ValueError(
            "malicious_fraction selects zero clients; increase the fraction or num_users"
        )

    rng = random.Random(trial_seed)
    return frozenset(rng.sample(range(1, num_users + 1), malicious_count))


def count_source_records(
    dataset: Dataset, indices: Iterable[int], source_class: int
) -> int:
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return sum(_label_as_int(targets[index]) == source_class for index in indices)

    count = 0
    for index in indices:
        _, label = dataset[index]
        count += _label_as_int(label) == source_class
    return count


def _label_as_int(label) -> int:
    if torch.is_tensor(label):
        return int(label.item())
    return int(label)


def _replace(label, source_class, target_class):
    if _label_as_int(label) != source_class:
        return label
    if torch.is_tensor(label):
        return label.new_tensor(target_class)
    return target_class

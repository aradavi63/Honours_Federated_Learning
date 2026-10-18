"""Load training-record exclusions shared by training and attack experiments."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def exclusion_file(dataset, supplied=None):
    path = supplied
    if path is None and dataset == "MNIST":
        path = ROOT / "training/resources/mnist_reserved_validation_indices.json"
    if path is None:
        return None
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Reserved validation manifest missing: {path}. Supply --exclude-indices."
        )
    return str(path)


def load_exclusions(path, dataset, size):
    if path is None:
        return set()
    payload = json.loads(Path(path).read_text())
    if payload.get("dataset") != dataset or payload.get("split") != "train":
        raise ValueError(
            "Exclusion manifest must match the dataset and official train split"
        )
    values = payload["indices"]
    if not isinstance(values, list) or any(
        type(i) is not int or not 0 <= i < size for i in values
    ):
        raise ValueError("Invalid reserved training-record IDs")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate reserved IDs")
    return set(values)

"""Dataset cache paths for training and experiment entry points."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"


def dataset_directory(dataset, root=None):
    base = Path(root).expanduser().resolve() if root is not None else DEFAULT_DATA_ROOT
    return base / dataset.lower()

"""One directory convention for new training and attack experiments."""

from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = frozenset({"training", "clean", "gia", "membership", "poisoning"})


def default_results_root(experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    return ROOT / "results" / experiment


def trial_directory(results_root, dataset, seed):
    return Path(results_root).resolve() / dataset / f"trial_{seed}"


def portable_path(value):
    """Store repository paths without recording a user's home directory."""
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return "<external-path>"


def portable_options(value):
    """Make path-valued options safe for published configuration metadata."""
    values = asdict(value) if is_dataclass(value) else dict(value)
    for key in ("data_root", "results_root", "exclusion_file", "exclude_indices"):
        if key in values:
            values[key] = portable_path(values[key])
    return values


def portable_text(value):
    """Remove local roots from diagnostic strings stored in result files."""
    result = str(value)
    for root, replacement in ((str(ROOT.resolve()), "."), (str(Path.home()), "<home>")):
        result = result.replace(root, replacement)
        result = result.replace(root.replace("\\", "/"), replacement)
    return result


def poisoning_destination(trial, scheme, condition, mapping, clean_trial=None):
    """A clean control is shared across mappings within a scheme and trial."""
    if condition == "clean":
        return (
            Path(clean_trial) / scheme
            if clean_trial is not None
            else Path(trial) / scheme / "clean"
        )
    if condition == "attack":
        return Path(trial) / scheme / "attack" / mapping
    raise ValueError(f"Unknown poisoning condition: {condition}")

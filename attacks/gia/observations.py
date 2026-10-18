"""Utilities for constructing the two attacker observations used in the study."""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch


def subtract_states(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    """Subtract two state dictionaries while preserving parameter order."""
    if list(left) != list(right):
        raise ValueError(
            "State dictionaries do not contain the same ordered parameter names."
        )
    return OrderedDict((name, left[name] - right[name]) for name in left)


def state_to_list(
    state: Mapping[str, torch.Tensor], names: Sequence[str]
) -> list[torch.Tensor]:
    """Select state tensors in the order expected by the reconstructor."""
    return [state[name].detach().clone() for name in names]


def max_abs_difference(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> float:
    if list(left) != list(right):
        raise ValueError(
            "State dictionaries do not contain the same ordered parameter names."
        )
    return max((left[name] - right[name]).abs().max().item() for name in left)

"""Derive reproducible seeds for GIA initial images and FedMIA query gradients."""

from __future__ import annotations

import hashlib
import json


def derive_seed(trial_seed: int, namespace: str, *parts: object) -> int:
    """Derive a stable 63-bit seed independent of Python's hash salt."""
    payload = json.dumps(
        [int(trial_seed), str(namespace), *parts],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & (
        (1 << 63) - 1
    )

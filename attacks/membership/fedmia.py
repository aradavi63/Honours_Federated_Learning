"""Membership inference using FedMIA and global-model losses.

FedMIA compares each query's score across clients. Global loss averages its
negative loss across global models. Higher scores suggest membership in both.
"""

import numpy as np
import torch
from scipy.special import ndtr
from sklearn.metrics import roc_auc_score, roc_curve


def fedmia(measurements, target=0, filter_outliers=True):
    """Score queries against other clients; input is [round, client, query]."""
    values = np.asarray(measurements, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 3 or not np.isfinite(values).all():
        raise ValueError("FedMIA needs finite [round, >=3 clients, query] measurements")
    if (
        not 0 <= target < values.shape[1]
        or values.shape[0] == 0
        or values.shape[2] == 0
    ):
        raise ValueError("Invalid target client or empty measurement sequence")
    reference = np.delete(values, target, axis=1)
    mean = reference.mean(axis=1, keepdims=True)
    std = reference.std(axis=1, keepdims=True)
    # Exclude scores above mean + 3 std, keeping ties.
    keep = (
        reference <= mean + 3 * std
        if filter_outliers
        else np.ones_like(reference, bool)
    )
    count = keep.sum(axis=1)
    mu = (reference * keep).sum(axis=1) / count
    variance = (((reference - mu[:, None, :]) ** 2) * keep).sum(axis=1) / count
    per_round = ndtr((values[:, target] - mu) / np.sqrt(variance + 1e-8))
    return per_round.mean(axis=0)


def loss_series(negative_losses):
    """Average negative losses across global models; input is [checkpoint, query]."""
    values = np.asarray(negative_losses)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("Loss-Series needs finite [checkpoint, query] scores")
    return values.mean(axis=0)


def measurements(
    model,
    initial_state,
    released_states,
    images,
    labels,
    device,
    gradient_mode="train",
    compute_cosine=True,
):
    """Measure negative cross-entropy loss and optional gradient cosine similarity.

    Loss uses eval mode. Query gradients use gradient_mode, with fresh dropout
    in train mode. Measurements use a separate attacker model.
    """
    model.eval()
    images, labels = images.to(device), labels.to(device)
    losses = []
    with torch.no_grad():
        for state in released_states:
            model.load_state_dict(state)
            parts = [
                torch.nn.functional.cross_entropy(model(x), y, reduction="none")
                for x, y in zip(images.split(32), labels.split(32))
            ]
            losses.append(-torch.cat(parts).cpu().numpy())
    if not compute_cosine:
        return np.stack(losses), None
    model.load_state_dict(initial_state)
    if gradient_mode not in ("train", "eval"):
        raise ValueError("gradient_mode must be train or eval")
    model.train(gradient_mode == "train")
    names = [name for name, _ in model.named_parameters()]
    # Reverse the model change to match the gradient direction.
    updates = torch.stack(
        [
            torch.cat([(initial_state[n] - state[n]).flatten() for n in names])
            for state in released_states
        ]
    ).to(device)
    updates = torch.nn.functional.normalize(updates, dim=1)
    cosine = []
    for x, y in zip(images, labels):
        loss = torch.nn.functional.cross_entropy(model(x.unsqueeze(0)), y.unsqueeze(0))
        gradient = torch.autograd.grad(loss, tuple(model.parameters()))
        gradient = torch.cat([part.flatten() for part in gradient])
        cosine.append(
            (updates @ torch.nn.functional.normalize(gradient, dim=0))
            .detach()
            .cpu()
            .numpy()
        )
    return np.stack(losses), np.stack(cosine, axis=1)


def evaluate(scores, membership):
    truth = np.asarray(membership, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or set(truth) != {0, 1}:
        raise ValueError(
            "ROC evaluation requires finite scores and both membership classes"
        )
    fpr, tpr, _ = roc_curve(truth, scores, drop_intermediate=False)
    negatives = int((truth == 0).sum())
    return {
        "auc": float(roc_auc_score(truth, scores)),
        "members": int(truth.sum()),
        "nonmembers": negatives,
        "fpr_resolution": 1 / negatives,
        "tpr_at_1pct_fpr": float(tpr[fpr <= 0.01].max()) if negatives >= 100 else None,
        "tpr_at_0_1pct_fpr": float(tpr[fpr <= 0.001].max())
        if negatives >= 1000
        else None,
    }

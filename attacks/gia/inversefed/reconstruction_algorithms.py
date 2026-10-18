"""Focused OP-GIA/IG reconstruction routines for recorded FedAvg updates."""

from __future__ import annotations

import time
from collections import OrderedDict
from copy import deepcopy
from typing import Callable, Sequence

import torch
from torch.func import functional_call

from .metrics import total_variation

DEFAULT_CONFIG = {
    "signed": True,
    "boxed": True,
    "cost_fn": "sim",
    "indices": "def",
    "weights": "equal",
    "lr": 0.1,
    "optim": "adam",
    "restarts": 1,
    "max_iterations": 24_000,
    "total_variation": 1e-2,
    "init": "randn",
    "lr_decay": True,
    "scoring_choice": "loss",
    "log_interval": 500,
    "iterate_selection": "final",
}


class ReplayDropout(torch.nn.Module):
    """Replay the target client's recorded dropout masks during simulation."""

    def __init__(self, masks: Sequence[torch.Tensor]):
        super().__init__()
        self.masks = [mask.detach().clone() for mask in masks]
        self.position = 0

    def reset_replay(self) -> None:
        self.position = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return inputs
        if self.position >= len(self.masks):
            raise RuntimeError("The recorded dropout schedule was exhausted.")
        mask = self.masks[self.position].to(device=inputs.device, dtype=inputs.dtype)
        self.position += 1
        if mask.shape != inputs.shape:
            raise RuntimeError(
                f"Recorded dropout shape {tuple(mask.shape)} does not match "
                f"the simulated activation shape {tuple(inputs.shape)}."
            )
        return inputs * mask


def _validate_config(config: dict | None) -> dict:
    merged = DEFAULT_CONFIG.copy()
    if config:
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown reconstruction settings: {sorted(unknown)}")
        merged.update(config)
    if merged["iterate_selection"] not in ("final", "best_objective"):
        raise ValueError("Unknown iterate selection")
    return merged


def reconstruction_costs(
    trial_update: Sequence[torch.Tensor],
    observed_update: Sequence[torch.Tensor],
    cost_fn: str = "sim",
    indices: str | Sequence[int] = "def",
    weights: str = "equal",
) -> torch.Tensor:
    """Compare a simulated update with the observed client update."""
    if len(trial_update) != len(observed_update):
        raise ValueError("Simulated and observed updates have different lengths.")
    if weights != "equal":
        raise ValueError("Only equal parameter weights are used in this experiment.")
    if isinstance(indices, str):
        if indices == "def":
            chosen = list(range(len(observed_update)))
        elif indices == "top10":
            count = min(10, len(observed_update))
            norms = torch.stack([part.norm() for part in observed_update])
            chosen = torch.topk(norms, count).indices.tolist()
        else:
            raise ValueError(f"Unsupported parameter selection: {indices}")
    else:
        chosen = list(indices)

    if cost_fn == "sim":
        dot = trial_update[0].new_zeros(())
        trial_norm = trial_update[0].new_zeros(())
        observed_norm = trial_update[0].new_zeros(())
        for index in chosen:
            trial, observed = trial_update[index], observed_update[index]
            dot = dot + (trial * observed).sum()
            trial_norm = trial_norm + trial.square().sum()
            observed_norm = observed_norm + observed.square().sum()
        # Empty assumed Poisson draws can produce a zero simulated update.
        # Clamp before sqrt so its derivative stays finite at zero.
        denominator = (trial_norm * observed_norm).clamp_min(1e-24).sqrt()
        return 1.0 - dot / denominator

    cost = trial_update[0].new_zeros(())
    for index in chosen:
        difference = trial_update[index] - observed_update[index]
        if cost_fn == "l2":
            cost = cost + difference.square().sum()
        elif cost_fn == "l1":
            cost = cost + difference.abs().sum()
        else:
            raise ValueError(f"Unsupported reconstruction cost: {cost_fn}")
    return cost


def loss_steps(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_schedule: Sequence[Sequence[int]],
    lr: float,
    momentum: float = 0.5,
) -> list[torch.Tensor]:
    """Simulate local SGD with the supplied batches and momentum."""
    for module in model.modules():
        reset = getattr(module, "reset_replay", None)
        if reset is not None:
            reset()
    original = OrderedDict(model.named_parameters())
    parameters = OrderedDict(model.named_parameters())
    model_buffers = OrderedDict(model.named_buffers())
    velocities: OrderedDict[str, torch.Tensor | None] = OrderedDict(
        (name, None) for name in parameters
    )
    for batch_indices in batch_schedule:
        index = torch.as_tensor(batch_indices, dtype=torch.long, device=inputs.device)
        output = functional_call(
            model, (parameters, model_buffers), (inputs.index_select(0, index),)
        )
        loss = torch.nn.functional.cross_entropy(output, labels.index_select(0, index))
        gradients = torch.autograd.grad(
            loss, tuple(parameters.values()), create_graph=True
        )
        updated = OrderedDict()
        for (name, parameter), gradient in zip(parameters.items(), gradients):
            previous = velocities[name]
            velocity = gradient if previous is None else momentum * previous + gradient
            velocities[name] = velocity
            updated[name] = parameter - lr * velocity
        parameters = updated
    return [parameters[name] - original[name] for name in original]


class GradientReconstructor:
    """Signed-Adam image optimization retained from OP-GIA/IG."""

    def __init__(self, model, mean_std, config, num_images):
        self.model = deepcopy(model)
        self.model.train()
        self.mean_std = mean_std
        self.config = _validate_config(config)
        self.num_images = num_images
        first = next(self.model.parameters())
        self.device, self.dtype = first.device, first.dtype

    def _init_images(self, image_shape):
        shape = (self.config["restarts"], self.num_images, *image_shape)
        if self.config["init"] == "randn":
            return torch.randn(shape, device=self.device, dtype=self.dtype)
        if self.config["init"] == "rand":
            return torch.rand(shape, device=self.device, dtype=self.dtype) * 2 - 1
        raise ValueError(f"Unsupported initialization: {self.config['init']}")

    def reconstruct(self, observed_update, labels, image_shape, initial_images=None):
        if labels.shape[0] != self.num_images:
            raise ValueError("Known labels must contain one label per image.")
        observed = [
            part.to(device=self.device, dtype=self.dtype) for part in observed_update
        ]
        labels = labels.to(self.device)
        initial = (
            self._init_images(image_shape)
            if initial_images is None
            else initial_images.to(device=self.device, dtype=self.dtype).clone()
        )
        if tuple(initial.shape) != (
            self.config["restarts"],
            self.num_images,
            *image_shape,
        ):
            raise ValueError("Invalid supplied candidate shape")
        if (
            not all(torch.isfinite(p).all() for p in observed)
            or not torch.isfinite(initial).all()
        ):
            raise FloatingPointError("Nonfinite observation or initial candidate")
        candidates, scores, histories = [], [], []
        started = time.perf_counter()
        for restart in range(self.config["restarts"]):
            print(
                f"Starting reconstruction restart {restart + 1}/{self.config['restarts']}...",
                flush=True,
            )
            candidate, history = self._run_trial(
                initial[restart], observed, labels, restart + 1
            )
            scoring_candidate = candidate.detach().clone().requires_grad_(True)
            # Upstream scoring_choice='loss' ranks restarts by matching cost, not TV.
            score = (
                self._score_trial(scoring_candidate, observed, labels).detach().item()
            )
            if not torch.isfinite(torch.tensor(score)):
                raise FloatingPointError("Nonfinite restart score")
            candidates.append(candidate.detach())
            scores.append(score)
            histories.append(history)
        best = min(range(len(scores)), key=scores.__getitem__)
        return candidates[best], {
            "opt": scores[best],
            "restart": best,
            "runtime_seconds": time.perf_counter() - started,
            "history": histories[best],
            "histories": histories,
            "restart_scores": scores,
            "initial_images": initial[best].detach().cpu(),
            "initial_matching_cost": float(
                self._score_trial(
                    initial[best].detach().requires_grad_(True), observed, labels
                ).detach()
            ),
        }

    def _objective(self, candidate, observed_update, labels):
        raise NotImplementedError

    def _score_trial(self, candidate, observed_update, labels):
        objective = self._objective(candidate, observed_update, labels)
        if self.config["scoring_choice"] != "loss":
            raise ValueError(
                "The focused IG experiment supports matching-loss restart selection"
            )
        return objective - self.config["total_variation"] * total_variation(candidate)

    def _run_trial(self, candidate, observed_update, labels, restart_number):
        candidate = candidate.detach().clone().requires_grad_(True)
        if self.config["optim"] != "adam":
            raise ValueError(
                "The focused experiment supports OP-GIA's Adam optimizer only."
            )
        optimizer = torch.optim.Adam([candidate], lr=self.config["lr"])
        maximum = int(self.config["max_iterations"])
        scheduler = None
        if self.config["lr_decay"]:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[maximum // 2.667, maximum // 1.6, maximum // 1.142],
                gamma=0.1,
            )
        mean, std = self.mean_std
        mean = mean.to(device=self.device, dtype=self.dtype)
        std = std.to(device=self.device, dtype=self.dtype)
        history = []
        best_candidate = candidate.detach().clone()
        best_objective = float("inf")
        log_interval = max(1, int(self.config["log_interval"]))
        trial_started = time.perf_counter()

        for iteration in range(maximum):
            step_lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad(set_to_none=True)
            self.model.zero_grad(set_to_none=True)
            objective = self._objective(candidate, observed_update, labels)
            objective_value = float(objective.detach())
            if not torch.isfinite(objective):
                raise FloatingPointError(
                    f"Non-finite reconstruction objective at iteration {iteration}"
                )
            if objective_value < best_objective:
                best_objective = objective_value
                best_candidate = candidate.detach().clone()
            objective.backward()
            if candidate.grad is None or not torch.isfinite(candidate.grad).all():
                raise FloatingPointError(
                    f"Invalid image gradient at iteration {iteration}"
                )
            if self.config["signed"]:
                candidate.grad.sign_()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                if self.config["boxed"]:
                    candidate.copy_(
                        torch.maximum(
                            torch.minimum(candidate, (1 - mean) / std), -mean / std
                        )
                    )
                if iteration % log_interval == 0 or iteration + 1 == maximum:
                    completed = iteration + 1
                    elapsed = time.perf_counter() - trial_started
                    remaining = maximum - completed
                    eta = elapsed / completed * remaining if completed else 0.0
                    fraction = completed / maximum
                    bar_width = 28
                    filled = min(bar_width, int(fraction * bar_width))
                    bar = "#" * filled + "-" * (bar_width - filled)
                    history.append(
                        {
                            "iteration": iteration,
                            "objective": objective_value,
                            "learning_rate": step_lr,
                        }
                    )
                    print(
                        f"restart {restart_number} [{bar}] "
                        f"{completed}/{maximum} ({fraction * 100:5.1f}%) "
                        f"loss={objective.detach().item():.6f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                        flush=True,
                    )
        # The loop scores the pre-step candidate. Include the final updated image.
        final_objective = float(
            self._objective(candidate, observed_update, labels).detach()
        )
        if not torch.isfinite(torch.tensor(final_objective)):
            raise FloatingPointError("Non-finite final reconstruction objective")
        if final_objective < best_objective:
            best_candidate = candidate.detach().clone()
        return (
            candidate.detach().clone()
            if self.config["iterate_selection"] == "final"
            else best_candidate
        ), history


class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct images from a recorded plaintext FedAvg client update."""

    def __init__(
        self,
        model,
        mean_std,
        batch_schedule,
        local_lr,
        momentum,
        config,
        num_images,
        observation_transform: Callable | None = None,
    ):
        super().__init__(model, mean_std, config, num_images)
        self.batch_schedule = [list(batch) for batch in batch_schedule]
        self.local_lr = local_lr
        self.momentum = momentum
        self.observation_transform = observation_transform or (
            lambda tensors: list(tensors)
        )

    def _objective(self, candidate, observed_update, labels):
        simulated = loss_steps(
            self.model,
            candidate,
            labels,
            self.batch_schedule,
            self.local_lr,
            self.momentum,
        )
        simulated = self.observation_transform(simulated)
        objective = reconstruction_costs(
            simulated,
            observed_update,
            cost_fn=self.config["cost_fn"],
            indices=self.config["indices"],
            weights=self.config["weights"],
        )
        if self.config["total_variation"] > 0:
            objective = objective + self.config["total_variation"] * total_variation(
                candidate
            )
        return objective


class AggregateFedAvgReconstructor(GradientReconstructor):
    """Reconstruct all clients jointly from the server's averaged FedAvg update."""

    def __init__(
        self,
        model,
        mean_std,
        client_schedules,
        local_lr,
        momentum,
        config,
        num_images,
    ):
        super().__init__(model, mean_std, config, num_images)
        self.client_schedules = client_schedules
        self.local_lr = local_lr
        self.momentum = momentum

    def _objective(self, candidate, observed_update, labels):
        aggregate = [torch.zeros_like(part) for part in observed_update]
        for client in self.client_schedules:
            start, stop = int(client["start"]), int(client["stop"])
            local_update = loss_steps(
                self.model,
                candidate[start:stop],
                labels[start:stop],
                client["schedule"],
                self.local_lr,
                self.momentum,
            )
            aggregate = [total + local for total, local in zip(aggregate, local_update)]
        simulated = [part / len(self.client_schedules) for part in aggregate]
        objective = reconstruction_costs(
            simulated,
            observed_update,
            cost_fn=self.config["cost_fn"],
            indices=self.config["indices"],
            weights=self.config["weights"],
        )
        if self.config["total_variation"] > 0:
            objective = objective + self.config["total_variation"] * total_variation(
                candidate
            )
        return objective

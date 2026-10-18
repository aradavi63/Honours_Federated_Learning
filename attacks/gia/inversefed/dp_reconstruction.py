"""Differentiable clipping-aware attack surrogate; training still uses Opacus.

Matches Opacus 1.4 DPOptimizer's flat clipping and expected-batch denominator.
Unknown noise is omitted, not recovered or subtracted. Over multiple local steps
this is a surrogate trajectory, not the expectation of the noisy trajectory.
"""

from collections import OrderedDict

import torch
from torch.func import functional_call

from .metrics import total_variation
from .reconstruction_algorithms import GradientReconstructor, reconstruction_costs


def clipped_loss_steps(
    model, images, labels, schedule, lr, momentum, clip, expected_batch_size
):
    original = OrderedDict(model.named_parameters())
    params = OrderedDict(original)
    buffers = OrderedDict(model.named_buffers())
    velocity = {name: torch.zeros_like(p) for name, p in params.items()}
    for batch in schedule:
        gradients = [p * 0 for p in params.values()]
        for index in batch:
            output = functional_call(
                model, (params, buffers), (images[index : index + 1],)
            )
            loss = torch.nn.functional.cross_entropy(output, labels[index : index + 1])
            per_record = torch.autograd.grad(
                loss, tuple(params.values()), create_graph=True
            )
            norm = torch.stack([g.norm(2) for g in per_record]).norm(2)
            factor = (clip / (norm + 1e-6)).clamp(max=1.0)
            gradients = [s + factor * g for s, g in zip(gradients, per_record)]
        updated = OrderedDict()
        for (name, p), g in zip(params.items(), gradients):
            velocity[name] = momentum * velocity[name] + g / expected_batch_size
            updated[name] = p - lr * velocity[name]
        params = updated
    # Preserve an image-gradient path even if the independent assumed draws are all empty.
    return [params[name] - p + images.sum() * 0 for name, p in original.items()]


class DPFedAvgReconstructor(GradientReconstructor):
    def __init__(self, model, mean_std, view, config):
        super().__init__(model, mean_std, config, len(view["labels"]))
        self.view = view

    def _objective(self, candidate, observed_update, labels):
        v = self.view
        if "client_schedules" in v:
            per_client = [
                clipped_loss_steps(
                    self.model,
                    candidate[e["start"] : e["stop"]],
                    labels[e["start"] : e["stop"]],
                    e["schedule"],
                    v["local_lr"],
                    v["momentum"],
                    v["max_grad_norm"],
                    e["expected_batch_size"],
                )
                for e in v["client_schedules"]
            ]
            simulated = [torch.stack(parts).mean(0) for parts in zip(*per_client)]
        else:
            simulated = clipped_loss_steps(
                self.model,
                candidate,
                labels,
                v["batch_schedule"],
                v["local_lr"],
                v["momentum"],
                v["max_grad_norm"],
                v["expected_batch_size"],
            )
        return reconstruction_costs(
            simulated,
            observed_update,
            self.config["cost_fn"],
            self.config["indices"],
            self.config["weights"],
        ) + self.config["total_variation"] * total_variation(candidate)

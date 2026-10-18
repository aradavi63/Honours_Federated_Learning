"""Opacus DP-SGD; persistent per-client accounting across communication rounds."""

import copy
import math
import secrets
import warnings

import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from training.local import LocalUpdateCNN


def make_private_trainer(
    partitions, noise_multiplier, max_grad_norm, delta, secure_mode=False
):
    if (
        not all(math.isfinite(v) for v in (noise_multiplier, max_grad_norm, delta))
        or noise_multiplier <= 0
        or max_grad_norm <= 0
        or not 0 < delta < 1
    ):
        raise ValueError("DP noise/clip must be positive and delta in (0,1)")
    lookup = {
        tuple(map(int, indices)): client for client, indices in partitions.items()
    }
    engines = {}
    datasets = {}
    # Filter the Opacus hook deprecation before worker threads start;
    # warning filters are process-wide.
    warnings.filterwarnings(
        "ignore",
        message=r"Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated.*",
        category=UserWarning,
        module=r"torch\.nn\.modules\.module",
    )
    if not secure_mode:
        print(
            "DP-SGD: clipping and Gaussian noise enabled; experimental RNG (secure_mode=False).",
            flush=True,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Secure RNG turned off\..*",
            category=UserWarning,
            module=r"opacus\.privacy_engine",
        )

    class PrivateLocalUpdate(LocalUpdateCNN):
        def train(self, net):
            client = lookup[tuple(self.ldr_train.dataset.idxs)]
            current = self.ldr_train.dataset
            if client not in datasets:
                datasets[client] = current
            else:
                # Fixed client indices, stable identity for Opacus. Refresh the
                # backing view so round-dependent poisoning still takes effect.
                datasets[client].dataset = current.dataset
            if client not in engines:
                engines[client] = PrivacyEngine(
                    accountant="rdp", secure_mode=secure_mode
                )
            engine = engines[client]
            net.train()
            ModuleValidator.validate(net, strict=True)
            # Never derive DP noise or sampling randomness from the public experiment seed.
            sampling_rng = torch.Generator().manual_seed(secrets.randbits(63))
            loader = DataLoader(
                datasets[client],
                batch_size=self.args.local_bs,
                shuffle=True,
                generator=sampling_rng,
            )
            noise_rng = (
                None
                if secure_mode
                else torch.Generator(device=self.args.device).manual_seed(
                    secrets.randbits(63)
                )
            )
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=self.args.lr,
                momentum=getattr(self.args, "momentum", 0.5),
            )
            private, optimizer, loader = engine.make_private(
                module=net,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                poisson_sampling=True,
                noise_generator=noise_rng,
            )
            for _ in range(self.args.local_ep):
                for images, labels in loader:
                    optimizer.zero_grad()
                    empty = len(labels) == 0
                    if empty:
                        # The repository CNN's view(0,-1) cannot process an empty batch.
                        # A public zero dummy contributes exactly zero; still take a noise-only step.
                        images = torch.zeros((1, *images.shape[1:]), dtype=images.dtype)
                        labels = torch.zeros(1, dtype=torch.long)
                    loss = torch.nn.functional.cross_entropy(
                        private(images.to(self.args.device)),
                        labels.to(self.args.device),
                    )
                    (loss * 0 if empty else loss).backward()
                    optimizer.step()
            state = {
                name: value.detach().clone()
                for name, value in private._module.state_dict().items()
            }
            private.remove_hooks()
            # Never upload an unnoised private training loss alongside the DP model.
            return state, 0.0

    def report():
        return {
            str(client): {
                "epsilon": engine.get_epsilon(delta),
                "delta": delta,
                "history": copy.deepcopy(engine.accountant.history),
            }
            for client, engine in engines.items()
        }

    return PrivateLocalUpdate, report

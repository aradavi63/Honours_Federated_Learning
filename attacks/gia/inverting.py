"""Run OP-GIA/IG against a recorded plaintext or SI-HEFL observation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from attacks.gia.inversefed import AggregateFedAvgReconstructor, FedAvgReconstructor
from attacks.gia.inversefed.dp_reconstruction import DPFedAvgReconstructor
from attacks.gia.inversefed.metrics import cw_ssim, mse, psnr, ssim
from attacks.gia.inversefed.reconstruction_algorithms import ReplayDropout
from models.networks import Cifar10cnn, Mnistcnn
from training.randomness import derive_seed


def build_model(view: dict, device: torch.device) -> torch.nn.Module:
    if view["model"] == "Mnistcnn":
        model = Mnistcnn(SimpleNamespace(num_classes=10)).to(device)
    elif view["model"] == "Cifar10cnn":
        model = Cifar10cnn(SimpleNamespace(num_classes=10)).to(device)
    else:
        raise ValueError(f"Unsupported recorded model: {view['model']}")
    model.load_state_dict(view["model_state"])
    dropout_masks = view.get("dropout_masks", [])
    if dropout_masks:
        model.dropout = ReplayDropout(dropout_masks)
    elif view.get("dropout_knowledge") in (
        "unknown_expected_value",
        "unknown_identity_surrogate",
        "known_identity_positive_control",
    ) and hasattr(model, "dropout"):
        model.dropout = torch.nn.Identity()
    return model


def denormalize(
    images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (images * std + mean).clamp(0, 1)


def match_reconstructions(
    reconstructed: torch.Tensor,
    ground_truth: torch.Tensor,
    labels=None,
    client_groups=None,
) -> tuple[torch.Tensor, list[int]]:
    """Resolve the permutation ambiguity using evaluation images only."""
    count = reconstructed.shape[0]
    costs = np.empty((count, count), dtype=np.float64)
    for target in range(count):
        for candidate in range(count):
            if (
                labels is not None and int(labels[target]) != int(labels[candidate])
            ) or (
                client_groups is not None
                and client_groups[target] != client_groups[candidate]
            ):
                costs[target, candidate] = np.inf
                continue
            costs[target, candidate] = torch.mean(
                (ground_truth[target] - reconstructed[candidate]) ** 2
            ).item()
    _, assignment = linear_sum_assignment(costs)
    return reconstructed[
        torch.as_tensor(assignment, device=reconstructed.device)
    ], assignment.tolist()


def optional_lpips(images: torch.Tensor, ground_truth: torch.Tensor) -> float | None:
    if importlib.util.find_spec("lpips") is None:
        return None
    import lpips

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
        ground_truth = ground_truth.repeat(1, 3, 1, 1)
    if min(images.shape[-2:]) < 64:
        images = torch.nn.functional.interpolate(
            images, size=(64, 64), mode="bilinear", align_corners=False
        )
        ground_truth = torch.nn.functional.interpolate(
            ground_truth, size=(64, 64), mode="bilinear", align_corners=False
        )
    scorer = lpips.LPIPS(net="alex").to(images.device)
    with torch.no_grad():
        return float(scorer(images * 2 - 1, ground_truth * 2 - 1).mean())


def save_comparison(
    path: Path, ground_truth: torch.Tensor, reconstructed: torch.Tensor
) -> None:
    shown = min(ground_truth.shape[0], 12)
    figure, axes = plt.subplots(2, shown, figsize=(1.8 * shown, 3.8), squeeze=False)
    for index in range(shown):
        truth = ground_truth[index].cpu()
        result = reconstructed[index].cpu()
        if truth.shape[0] == 1:
            axes[0, index].imshow(truth.squeeze(0), cmap="gray", vmin=0, vmax=1)
            axes[1, index].imshow(result.squeeze(0), cmap="gray", vmin=0, vmax=1)
        else:
            axes[0, index].imshow(truth.permute(1, 2, 0))
            axes[1, index].imshow(result.permute(1, 2, 0))
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    figure.text(0.01, 0.75, "Original", rotation=90, va="center", fontsize=10)
    figure.text(0.01, 0.25, "Reconstructed", rotation=90, va="center", fontsize=10)
    figure.tight_layout(rect=(0.04, 0, 1, 1))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def initial_candidates(view, restarts, seed):
    """Same client candidate block across schemes and observation scopes."""
    clients = view.get(
        "client_schedules",
        [dict(client=view["target_client"], start=0, stop=len(view["labels"]))],
    )
    blocks = []
    for client in clients:
        generator = torch.Generator().manual_seed(
            derive_seed(seed, "gia_initial_images", client["client"])
        )
        blocks.append(
            torch.randn(
                (restarts, client["stop"] - client["start"], *view["image_shape"]),
                generator=generator,
            )
        )
    return torch.cat(blocks, dim=1)


def run_attack(attack_view, evaluation_target, output_directory, **kwargs):
    """Keep failures explicit and never overwrite an earlier reconstruction."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    return _run_attack(attack_view, evaluation_target, output_directory, **kwargs)


def load_observation(value, condition=None):
    """Read a single observation or select one from a saved scheme bundle."""
    payload = (
        value
        if isinstance(value, dict)
        else torch.load(value, map_location="cpu", weights_only=True)
    )
    if "model_state" in payload or "images" in payload:
        return payload
    if condition is None:
        if len(payload) != 1:
            raise ValueError(
                "Choose --condition for a bundle with multiple observations"
            )
        condition = next(iter(payload))
    if condition not in payload:
        raise ValueError(f"Unknown observation condition: {condition}")
    return payload[condition]


def _run_attack(
    attack_view: dict | Path,
    evaluation_target: dict | Path,
    output_directory: Path,
    iterations: int = 24_000,
    restarts: int = 1,
    attack_lr: float = 0.1,
    total_variation: float = 1e-2,
    cost_fn: str = "sim",
    device_name: str = "cpu",
    attack_seed: int = 4010,
    progress_interval: int = 100,
    iterate_selection: str = "final",
    condition: str | None = None,
) -> dict:
    if (
        min(iterations, restarts) < 1
        or not np.isfinite([attack_lr, total_variation]).all()
        or attack_lr <= 0
        or total_variation < 0
    ):
        raise ValueError("Invalid reconstruction budget, learning rate or TV")
    random.seed(attack_seed)
    np.random.seed(attack_seed)
    torch.manual_seed(attack_seed)
    device = torch.device(device_name)
    view = load_observation(attack_view, condition)
    output_directory.mkdir(parents=True, exist_ok=True)

    model = build_model(view, device)
    channels = len(view["mean"])
    mean = torch.as_tensor(view["mean"], device=device).reshape(1, channels, 1, 1)
    std = torch.as_tensor(view["std"], device=device).reshape(1, channels, 1, 1)
    config = {
        "signed": True,
        "boxed": True,
        "cost_fn": cost_fn,
        "indices": "def",
        "weights": "equal",
        "lr": attack_lr,
        "optim": "adam",
        "restarts": restarts,
        "max_iterations": iterations,
        "total_variation": total_variation,
        "init": "randn",
        "lr_decay": True,
        "scoring_choice": "loss",
        "log_interval": progress_interval,
        "iterate_selection": iterate_selection,
    }
    print(
        f"Starting {view['condition']} with {iterations} iterations, "
        f"{restarts} restart(s), and {len(view['labels'])} image(s).",
        flush=True,
    )
    if (
        view.get("reconstruction_model")
        == "clipped_SGD_surrogate_without_realised_noise"
    ):
        reconstructor = DPFedAvgReconstructor(model, (mean, std), view, config)
    elif "client_schedules" in view:
        reconstructor = AggregateFedAvgReconstructor(
            model=model,
            mean_std=(mean, std),
            client_schedules=view["client_schedules"],
            local_lr=view["local_lr"],
            momentum=view["momentum"],
            config=config,
            num_images=len(view["labels"]),
        )
    else:
        reconstructor = FedAvgReconstructor(
            model=model,
            mean_std=(mean, std),
            batch_schedule=view["batch_schedule"],
            local_lr=view["local_lr"],
            momentum=view["momentum"],
            config=config,
            num_images=len(view["labels"]),
        )
    reconstructed, stats = reconstructor.reconstruct(
        view["observed_update"],
        view["labels"],
        view["image_shape"],
        initial_images=initial_candidates(view, restarts, attack_seed),
    )
    evaluation_started = time.perf_counter()
    evaluation = load_observation(evaluation_target, view["condition"])
    reconstructed_pixels = denormalize(reconstructed, mean, std)
    ground_truth = evaluation["images"].to(device)
    ground_truth_pixels = denormalize(ground_truth, mean, std)
    client_groups = None
    if "client_schedules" in view:
        client_groups = [None] * len(view["labels"])
        for client in view["client_schedules"]:
            client_groups[client["start"] : client["stop"]] = [client["client"]] * (
                client["stop"] - client["start"]
            )
    reconstructed_pixels, assignment = match_reconstructions(
        reconstructed_pixels, ground_truth_pixels, view["labels"], client_groups
    )

    if client_groups is None:
        target_positions = list(range(len(view["labels"])))
    else:
        target_positions = [
            index
            for index, client in enumerate(client_groups)
            if client == view["target_client"]
        ]
    if not target_positions:
        raise ValueError("The configured target client has no reconstructed images")
    target_positions = torch.tensor(target_positions, device=device)
    target_reconstruction = reconstructed_pixels.index_select(0, target_positions)
    target_ground_truth = ground_truth_pixels.index_select(0, target_positions)

    initial_pixels, _ = match_reconstructions(
        denormalize(stats["initial_images"].to(device), mean, std),
        ground_truth_pixels,
        view["labels"],
        client_groups,
    )
    per_image = []
    for i in range(len(view["labels"])):
        per_image.append(
            dict(
                position=i,
                label=int(view["labels"][i]),
                client=client_groups[i] if client_groups else view["target_client"],
                mse=mse(
                    reconstructed_pixels[i : i + 1], ground_truth_pixels[i : i + 1]
                ),
                initial_mse=mse(
                    initial_pixels[i : i + 1], ground_truth_pixels[i : i + 1]
                ),
                psnr=psnr(
                    reconstructed_pixels[i : i + 1], ground_truth_pixels[i : i + 1]
                ),
            )
        )
    with (output_directory / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)

    lpips_error = None
    try:
        lpips_value = optional_lpips(reconstructed_pixels, ground_truth_pixels)
    except (OSError, RuntimeError, ImportError) as error:
        lpips_value = None
        lpips_error = f"{type(error).__name__}: {error}"
        warnings.warn(f"LPIPS unavailable; preserving other GIA results: {lpips_error}")
    metric_values = {
        "condition": view["condition"],
        "scheme": view.get("scheme"),
        "source": view.get("source", "legacy_unspecified"),
        "primary_comparison": view.get("primary_comparison", False),
        "observation_scope": view.get("observation_scope"),
        "simulation": view.get("simulation", "strong"),
        "dataset": view.get("dataset", "MNIST"),
        "trial_seed": view["trial_seed"],
        "round": view["round"],
        "target_client": view["target_client"],
        "num_images": len(view["labels"]),
        "num_target_images": len(target_positions),
        "objective": stats["opt"],
        "initial_matching_cost": stats["initial_matching_cost"],
        "initial_mse": mse(initial_pixels, ground_truth_pixels),
        "black_image_mse": mse(
            torch.zeros_like(ground_truth_pixels), ground_truth_pixels
        ),
        "black_target_mse": mse(
            torch.zeros_like(target_ground_truth), target_ground_truth
        ),
        "initial_target_mse": mse(
            initial_pixels.index_select(0, target_positions), target_ground_truth
        ),
        "reconstruction_model": view.get(
            "reconstruction_model", "ordinary_SGD_surrogate"
        ),
        "iterate_selection": iterate_selection,
        "observed_update_l2": float(
            torch.cat([p.flatten() for p in view["observed_update"]]).norm()
        ),
        "mse": mse(reconstructed_pixels, ground_truth_pixels),
        "target_mse": mse(target_reconstruction, target_ground_truth),
        "psnr": psnr(reconstructed_pixels, ground_truth_pixels),
        "ssim_global": ssim(reconstructed_pixels, ground_truth_pixels),
        "cw_ssim": cw_ssim(reconstructed_pixels, ground_truth_pixels),
        "lpips": lpips_value,
        "runtime_seconds": stats["runtime_seconds"],
    }
    save_comparison(
        output_directory / "comparison.png",
        ground_truth_pixels.detach().cpu(),
        reconstructed_pixels.detach().cpu(),
    )
    metric_values["evaluation_seconds"] = time.perf_counter() - evaluation_started
    torch.save(
        {
            "originals": ground_truth_pixels.detach().cpu(),
            "reconstructed": reconstructed_pixels.detach().cpu(),
            "initial": initial_pixels.detach().cpu(),
            "labels": view["labels"].detach().cpu(),
            "clients": client_groups or [view["target_client"]] * len(view["labels"]),
            "metrics": metric_values,
            "optimization": {
                "selected_restart": stats["restart"],
                "histories": stats["histories"],
                "restart_scores": stats["restart_scores"],
                "parameters": config,
                "seed": attack_seed,
            },
            "assumptions": {
                "observation_scope": view["observation_scope"],
                "release_boundary": view.get("release_boundary"),
                "label_knowledge": view.get("label_knowledge"),
                "schedule_knowledge": view.get("schedule_knowledge"),
                "dropout_knowledge": view.get("dropout_knowledge"),
                "noise_knowledge": view.get("noise_knowledge"),
                "lpips_error": lpips_error,
            },
        },
        output_directory / "reconstruction.pt",
    )
    print(
        f"{view['condition']}: MSE={metric_values['mse']:.6f}, "
        f"PSNR={metric_values['psnr']:.3f}, "
        f"SSIM={metric_values['ssim_global']:.4f}."
    )
    return metric_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inverting recorded SI-HEFL/FedAvg client updates")
    parser.add_argument("--attack-view", type=Path, required=True)
    parser.add_argument("--evaluation-target", type=Path, required=True)
    parser.add_argument(
        "--condition", help="Observation name within a saved scheme bundle"
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--iter", type=int, default=24_000)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads for the standalone attack; the experiment runner also defaults to one",
    )
    parser.add_argument(
        "--iterate-selection", choices=["final", "best_objective"], default="final"
    )
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--total-variation", type=float, default=1e-2)
    parser.add_argument("--cost-fn", choices=["sim", "l2", "l1"], default="sim")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=4010)
    parser.add_argument(
        "--progress-interval",
        "--log-interval",
        dest="progress_interval",
        type=int,
        default=100,
        help="iterations between terminal progress updates",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.torch_threads < 1:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(arguments.torch_threads)
    run_attack(
        attack_view=arguments.attack_view,
        evaluation_target=arguments.evaluation_target,
        output_directory=arguments.output_directory,
        iterations=arguments.iter,
        restarts=arguments.restarts,
        attack_lr=arguments.lr,
        total_variation=arguments.total_variation,
        cost_fn=arguments.cost_fn,
        device_name=arguments.device,
        attack_seed=arguments.seed,
        progress_interval=arguments.progress_interval,
        iterate_selection=arguments.iterate_selection,
        condition=arguments.condition,
    )

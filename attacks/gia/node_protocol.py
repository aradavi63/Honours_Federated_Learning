"""Capture native server releases for GIA; keep synthetic controls separate."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from attacks.gia.inversefed.reconstruction_algorithms import loss_steps
from attacks.gia.observations import state_to_list, subtract_states
from schemes import ENCRYPTED_SCHEMES, PRIVATE_SCHEMES
from training.runner import run_training
from training.setup import DATASET_DETAILS, TrainingConfig, _build_model


@dataclass(frozen=True)
class NodeProtocolConfig(TrainingConfig):
    attack_round: int = 1
    target_client: int = 1


def cyclic_schedule(sample_count, local_epochs, batch_size):
    """Assumed sequential order; the native client's shuffle is not disclosed."""
    nbatches = max(1, math.ceil(sample_count / batch_size))
    schedule = []
    for _ in range(local_epochs):
        for b in range(nbatches):
            batch = list(range(b * batch_size, min((b + 1) * batch_size, sample_count)))
            if batch:
                schedule.append(batch)
    return schedule


def make_private_schedule(count, epochs, batch_size, seed):
    # Opacus PrivacyEngine: sample_rate=1/len(original_loader), expected=int(N*q).
    steps = math.ceil(count / batch_size)
    rate = 1 / steps
    rng = np.random.default_rng(seed)
    return [
        np.flatnonzero(rng.random(count) < rate).tolist() for _ in range(epochs * steps)
    ], int(count * rate)


def _controlled_update(
    start_state, images, labels, schedules, local_lr, momentum, config, aggregate
):
    """Compute the observed update as the attacker's own deterministic simulation."""
    devices = (
        [torch.device(config.device).index or 0]
        if str(config.device).startswith("cuda")
        else []
    )
    with torch.random.fork_rng(devices=devices):
        model = _build_model(config).to("cpu")
    model.load_state_dict(start_state)
    model.train()
    if hasattr(model, "dropout"):
        model.dropout = torch.nn.Identity()
    images = torch.stack(images)
    labels = torch.tensor(labels)
    if aggregate:
        per_client = []
        for entry in schedules:
            client_images = images[entry["start"] : entry["stop"]]
            client_labels = labels[entry["start"] : entry["stop"]]
            per_client.append(
                loss_steps(
                    model,
                    client_images,
                    client_labels,
                    entry["schedule"],
                    local_lr,
                    momentum,
                )
            )
        return [
            torch.stack([client[i] for client in per_client]).mean(0).detach()
            for i in range(len(per_client[0]))
        ]
    deltas = loss_steps(
        model, images, labels, schedules[0]["schedule"], local_lr, momentum
    )
    return [delta.detach() for delta in deltas]


def run_node_protocol(
    scheme, options, config, train, test, partitions, initial, output
):
    """Build attack views from the scheme's server-visible releases."""
    if not 1 <= config.attack_round <= config.rounds:
        raise ValueError("attack-round must lie within training rounds")
    if config.target_client not in partitions:
        raise ValueError("Unknown target client")
    paths = {"observations": []}
    # Constructing the name list must not advance training randomness.
    devices = (
        [torch.device(config.device).index or 0]
        if str(config.device).startswith("cuda")
        else []
    )
    with torch.random.fork_rng(devices=devices):
        names = [name for name, _ in _build_model(config).named_parameters()]

    def record(round_number, start, local, global_state):
        if round_number != config.attack_round:
            return
        if scheme == "fedavg":
            specifications = [
                dict(
                    condition="fedavg_client_update",
                    state=local[config.target_client],
                    clients=[config.target_client],
                    observation="server_received_client_model",
                ),
                dict(
                    condition="fedavg_server_aggregate",
                    state=global_state,
                    clients=list(partitions),
                    observation="server_computed_global_model",
                ),
            ]
            if getattr(options, "positive_control", False):
                specifications.append(
                    dict(
                        condition="positive_control_deterministic",
                        clients=[config.target_client],
                        observation="synthetic_known_schedule_update",
                        positive_control=True,
                    )
                )
        elif scheme in ENCRYPTED_SCHEMES:
            specifications = [
                dict(
                    condition=f"{scheme}_server_aggregate",
                    state=global_state,
                    clients=list(partitions),
                    observation="server_decrypted_aggregate_after_mask_removal",
                ),
            ]
        elif scheme == "dp_fedavg":
            specifications = [
                dict(
                    condition="dp_fedavg_client_update",
                    state=local[config.target_client],
                    clients=[config.target_client],
                    observation="server_received_noisy_client_update",
                ),
                dict(
                    condition="dp_fedavg_server_aggregate",
                    state=global_state,
                    clients=list(partitions),
                    observation="server_computed_noisy_global_aggregate",
                ),
            ]

        simulation = getattr(options, "simulation", "strong")
        if simulation not in ("strong", "weak"):
            raise ValueError(f"Unknown simulation level: {simulation}")

        for specification in specifications:
            aggregate = len(specification["clients"]) > 1
            details = DATASET_DETAILS[config.dataset]
            positive_control = specification.get("positive_control", False)
            # Candidate grouping and labels are oracle information, not protocol captures.
            labels, images, victim_schedules, indices = [], [], [], []
            offset = 0
            for client in specification["clients"]:
                values = [train[i] for i in partitions[client]]
                images.extend(x for x, _ in values)
                labels.extend(int(y) for _, y in values)
                indices.extend(partitions[client])
                victim_schedules.append(
                    dict(
                        client=client,
                        start=offset,
                        stop=offset + len(values),
                        schedule=cyclic_schedule(
                            len(values), config.local_epochs, config.local_batch_size
                        ),
                    )
                )
                offset += len(values)

            is_dp = scheme in PRIVATE_SCHEMES
            if positive_control:
                observed_update = _controlled_update(
                    start,
                    images,
                    labels,
                    victim_schedules,
                    config.local_lr,
                    config.momentum,
                    config,
                    aggregate,
                )
            else:
                observed_update = state_to_list(
                    subtract_states(specification["state"], start), names
                )

            # Knowing hyperparameters does not reveal the native private batch order.
            if simulation == "weak" and not is_dp and not positive_control:
                sim_bs, sim_epochs, sim_lr = 8, 2, 0.001
                attacker_lr = sim_lr
                attacker_schedules = [
                    dict(
                        client=e["client"],
                        start=e["start"],
                        stop=e["stop"],
                        schedule=cyclic_schedule(
                            e["stop"] - e["start"], sim_epochs, sim_bs
                        ),
                    )
                    for e in victim_schedules
                ]
            else:
                attacker_lr = config.local_lr
                attacker_schedules = victim_schedules

            view = dict(
                schema_version=9,
                scheme=scheme,
                condition=specification["condition"],
                source="synthetic_positive_control"
                if positive_control
                else "actual_server_observation",
                primary_comparison=not positive_control,
                release_boundary=(
                    None
                    if positive_control
                    else "ServerNodeClass.receiveMessage:RESULTS[0]"
                    if not aggregate
                    else "ServerNodeClass.decryptWeights:after_mask_removal"
                    if scheme in ENCRYPTED_SCHEMES
                    else "ServerNodeClass.trainingProcess:after_FedAvg_model_load"
                ),
                trial_seed=config.trial_seed,
                round=round_number,
                dataset=config.dataset,
                model=details["model"],
                model_state=start,
                image_shape=details["shape"],
                mean=details["mean"],
                std=details["std"],
                local_lr=attacker_lr,
                momentum=config.momentum,
                local_epochs=config.local_epochs,
                local_batch_size=config.local_batch_size,
                labels=torch.tensor(labels),
                target_client=config.target_client,
                observed_update=observed_update,
                observation=specification["observation"],
                observation_scope="aggregate" if aggregate else "individual_client",
                attacker="honest_but_curious_server",
                simulation="positive_control" if positive_control else simulation,
                victim_local_lr=config.local_lr,
                victim_local_batch_size=config.local_batch_size,
                label_knowledge="oracle_labels_and_client_counts; not inferred from protocol",
                schedule_knowledge=(
                    "deterministic_cyclic_order; synthetic_victim_equals_attacker"
                    if positive_control
                    else "assumed_cyclic_order; native_shuffle_unknown"
                    if not is_dp and simulation == "strong"
                    else "mismatched_hyperparameters_and_assumed_cyclic_order"
                    if not is_dp
                    else "independent_poisson_draws; actual_sampling_unknown"
                ),
                reconstruction_model="ordinary_SGD_surrogate",
                update_convention="local_update_at_round_start_model",
                parameter_names=names,
                dropout_knowledge=(
                    "known_identity_positive_control"
                    if positive_control
                    else "unknown_identity_surrogate"
                )
                if config.dataset == "CIFAR10"
                else "not_applicable",
            )
            if aggregate:
                view["client_schedules"] = attacker_schedules
            else:
                view["batch_schedule"] = attacker_schedules[0]["schedule"]
            if is_dp:
                dp_schedules = []
                for entry in attacker_schedules:
                    schedule, expected = make_private_schedule(
                        entry["stop"] - entry["start"],
                        config.local_epochs,
                        config.local_batch_size,
                        config.trial_seed + 900001 + entry["client"],
                    )
                    dp_schedules.append(
                        dict(entry, schedule=schedule, expected_batch_size=expected)
                    )
                if aggregate:
                    view["client_schedules"] = dp_schedules
                else:
                    view.update(
                        batch_schedule=dp_schedules[0]["schedule"],
                        expected_batch_size=dp_schedules[0]["expected_batch_size"],
                    )
                view.update(
                    max_grad_norm=options.max_grad_norm,
                    noise_multiplier=options.noise_multiplier,
                    noise_knowledge="distribution_only; realised_noise_unknown",
                    reconstruction_model="clipped_SGD_surrogate_without_realised_noise",
                )

            if any(
                key in view
                for key in ("images", "dataset_indices", "evaluation_target")
            ):
                raise RuntimeError("Private evaluator data entered the attacker view")
            paths["observations"].append(
                dict(
                    condition=specification["condition"],
                    attack_view=view,
                    evaluation_target=dict(
                        images=torch.stack(images),
                        labels=torch.tensor(labels),
                        dataset_indices=indices,
                        target_client=config.target_client,
                    ),
                )
            )

        # Preserve the primary observation keys used by older callers.
        paths.update(paths["observations"][0])

    run_training(
        scheme,
        options,
        config,
        train,
        test,
        partitions,
        initial,
        output,
        observer=record,
        experiment_name="gia",
        release_rounds={config.attack_round},
    )
    if not paths["observations"]:
        raise RuntimeError("Attack observation was not captured")
    torch.save(
        {item["condition"]: item["attack_view"] for item in paths["observations"]},
        Path(output) / "observations.pt",
    )
    torch.save(
        {
            item["condition"]: item["evaluation_target"]
            for item in paths["observations"]
        },
        Path(output) / "evaluation_targets.pt",
    )
    return paths

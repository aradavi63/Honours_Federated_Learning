"""Evaluate membership inference from the server-visible training releases."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from attacks.membership.fedmia import evaluate, fedmia, loss_series, measurements
from evaluation.paths import (
    default_results_root,
    trial_directory,
)
from evaluation.reporting import write_csv
from training.randomness import derive_seed
from training.runner import run_training
from training.setup import (
    TrainingConfig,
    _build_model,
    _build_node_args,
    _clone_state,
    _load_dataset,
    _sample_partitions,
    seed_everything,
)


def prepare_queries(dataset, partitions, target, count, seed, excluded=()):
    """Same-source held-out records, class matched; separate other-client negatives."""
    flat = [i for indices in partitions.values() for i in indices]
    if len(flat) != len(set(flat)):
        raise ValueError("FedMIA requires disjoint client datasets")
    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset.targets)
    member = rng.choice(
        partitions[target], min(count, len(partitions[target])), replace=False
    ).tolist()
    used = set(flat)
    if used.intersection(excluded):
        raise ValueError(
            "Membership training partitions contain reserved validation records"
        )
    used.update(excluded)
    available = {}
    for label in set(labels[member]):
        available[int(label)] = rng.permutation(
            [int(i) for i in np.where(labels == label)[0] if i not in used]
        ).tolist()
    nonmember = []
    for index in member:
        pool = available[int(labels[index])]
        if not pool:
            raise ValueError("Not enough disjoint class-matched held-out candidates")
        nonmember.append(pool.pop())
    other_pool = [
        i for client, indices in partitions.items() if client != target for i in indices
    ]
    other = []
    for index in member:
        matching = [i for i in other_pool if labels[i] == labels[index]]
        if matching:
            selected = int(rng.choice(matching))
            other.append(selected)
            other_pool.remove(selected)
    indices = member + nonmember + other
    groups = (
        ["target_member"] * len(member)
        + ["heldout"] * len(nonmember)
        + ["other_client"] * len(other)
    )
    images = torch.stack([dataset[i][0] for i in indices])
    query_labels = torch.tensor(labels[indices], dtype=torch.long)
    return images, query_labels, indices, groups


def evaluation_mask(groups, labels, negative_group):
    """Select class-matched positives for the other-client attribution check."""
    groups = np.asarray(groups)
    keep = groups == negative_group
    members = np.flatnonzero(groups == "target_member")
    if negative_group == "heldout":
        keep[members] = True
    elif negative_group == "other_client":
        labels = np.asarray(labels)
        remaining = {
            int(label): int(np.sum(keep & (labels == label)))
            for label in np.unique(labels[keep])
        }
        for index in members:
            label = int(labels[index])
            if remaining.get(label, 0):
                keep[index] = True
                remaining[label] -= 1
    else:
        raise ValueError(f"Unknown negative group: {negative_group}")
    return keep


def run_scheme(
    scheme, options, config, train, test, partitions, initial, queries, output
):
    images, labels, indices, groups = queries
    devices = (
        [torch.device(config.device).index or 0]
        if str(config.device).startswith("cuda")
        else []
    )
    with torch.random.fork_rng(devices=devices):
        attacker_model = _build_model(config)
    attacker_model.load_state_dict(initial)
    loss_history, cosine_history, round_ids = [], [], []
    observed_client_counts = []

    def record_round(round_number, starting_state, client_states, global_state):
        if round_number % options.attack_every == 0 or round_number == config.rounds:
            if client_states and set(client_states) != set(partitions):
                raise ValueError(
                    "Incomplete client releases cannot be used to fit FedMIA"
                )
            observed = [client_states[c] for c in partitions] if client_states else []
            observed_client_counts.append(len(observed))
            observed.append(global_state)
            started = time.perf_counter()
            devices = (
                [torch.device(config.device).index or 0]
                if str(config.device).startswith("cuda")
                else []
            )
            with torch.random.fork_rng(devices=devices):
                measurement_seed = derive_seed(
                    config.trial_seed, "membership_query_gradient", round_number
                )
                torch.manual_seed(measurement_seed)
                if devices:
                    torch.cuda.manual_seed_all(measurement_seed)
                # FedMIA calibration needs the same position-aligned queries for every client.
                loss_measure, cos_measure = measurements(
                    attacker_model,
                    starting_state,
                    observed,
                    images,
                    labels,
                    config.device,
                    gradient_mode=getattr(options, "query_gradient_mode", "train"),
                    compute_cosine=bool(client_states),
                )
            loss_history.append(loss_measure)
            if cos_measure is not None:
                cosine_history.append(cos_measure)
            round_ids.append(round_number)
            duration = time.perf_counter() - started
            print(
                f"{scheme}: recorded MIA round {round_number}, {len(indices)} queries, {duration:.1f}s",
                flush=True,
            )

    run_training(
        scheme,
        options,
        config,
        train,
        test,
        partitions,
        initial,
        output,
        observer=record_round,
        experiment_name="membership",
        release_rounds={
            r
            for r in range(1, config.rounds + 1)
            if r % options.attack_every == 0 or r == config.rounds
        },
    )
    losses = np.stack(loss_history)
    cosines = np.stack(cosine_history) if cosine_history else None
    observations = {
        "rounds": np.asarray(round_ids, dtype=np.int64),
        "clients": np.asarray(
            [*partitions, 0] if all(observed_client_counts) else [0], dtype=np.int64
        ),
        "target_client": np.asarray(options.target_client, dtype=np.int64),
        "dataset_indices": np.asarray(indices, dtype=np.int64),
        "labels": labels.cpu().numpy(),
        "groups": np.asarray(groups, dtype=str),
        "loss": losses,
    }
    if cosines is not None:
        observations["cosine"] = cosines
    np.savez_compressed(output / "observations.npz", **observations)

    # Global-model Loss-Series is our adaptation; positives come from the target client.
    global_model_scores = {
        "GM-Loss-Series": loss_series(losses[:, -1]),
    }
    target_client_scores = {}
    individual_available = bool(observed_client_counts) and all(
        count == len(partitions) for count in observed_client_counts
    )
    if individual_available:
        target = list(partitions).index(options.target_client)
        target_client_scores = {
            "FedMIA-I": fedmia(losses[:, :-1], target, filter_outliers=False),
            "FedMIA-II": fedmia(cosines[:, :-1], target),
        }
    groups = np.asarray(groups)
    score_map = {**target_client_scores, **global_model_scores}
    write_csv(
        output / "scores.csv",
        [
            dict(
                query=q,
                dataset_index=int(indices[q]),
                label=int(labels[q]),
                group=groups[q],
                **{name: float(values[q]) for name, values in score_map.items()},
            )
            for q in range(len(indices))
        ],
    )
    rows = []

    def evaluate_family(score_map, destination, family, unit, negative_groups):
        for name, values in score_map.items():
            for negative_group in negative_groups:
                if not np.any(groups == negative_group):
                    continue
                keep = evaluation_mask(groups, labels.numpy(), negative_group)
                result = evaluate(values[keep], groups[keep] == "target_member")
                destination.append(
                    {
                        "scheme": scheme,
                        "attack": name,
                        "family": family,
                        "membership_unit": unit,
                        "negative_group": negative_group,
                        **result,
                    }
                )

    evaluate_family(
        target_client_scores,
        rows,
        "target_client_fedmia",
        "target_client",
        ("heldout", "other_client"),
    )
    evaluate_family(
        global_model_scores, rows, "global_model", "global_training_union", ("heldout",)
    )
    reason = (
        "server observer received every individual client release"
        if individual_available
        else "individual target and non-target client releases are absent from the evaluated SIHEFL server observation"
    )
    availability = {
        "threat_model": "passive_non_colluding_server",
        "observation_rounds": round_ids,
        "candidate_fingerprint": _candidate_fingerprint(images, labels, indices),
        "global_positive_population": "target-client members sampled from federation training union",
        "individual_client_releases_observed_per_round": observed_client_counts,
        "attacks": {
            "FedMIA-I": {
                "status": "available" if individual_available else "unavailable",
                "membership_unit": "target_client",
                "requires": "individual client models",
                "reason": reason,
            },
            "FedMIA-II": {
                "status": "available" if individual_available else "unavailable",
                "membership_unit": "target_client",
                "requires": "individual client updates",
                "reason": reason,
            },
            "GM-Loss-Series": {
                "status": "available",
                "membership_unit": "global_training_union",
                "requires": "broadcast global model",
                "reason": "the server always observes the released global model",
            },
        },
        "qualification": (
            "The captured SIHEFL model-release view supplies no individual client models. "
            "This is not a complete transcript/key/collusion audit. GM-Loss-Series tests "
            "union membership for candidates sampled from the designated target client; "
            "it does not establish target-client attribution."
        ),
    }
    configuration = output / "configuration.json"
    payload = json.loads(configuration.read_text(encoding="utf-8"))
    payload["membership"] = availability
    configuration.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return rows


def _candidate_fingerprint(images, labels, indices):
    """Verify paired inputs without passing membership truth to the scorer."""
    import hashlib

    digest = hashlib.sha256(json.dumps(list(map(int, indices))).encode())
    for value in (images, labels):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-indices", default=None)
    parser.add_argument(
        "--query-gradient-mode",
        choices=["train", "eval"],
        default="train",
        help="Upstream FedMIA uses train mode; eval is a deterministic variant",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="Local SGD momentum",
    )
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset cache root (default: repository/data)",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["fedavg", "sihefl", "dp_fedavg", "dp_sihefl"],
        default=["fedavg", "sihefl", "dp_fedavg"],
    )
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10"], default="MNIST")
    parser.add_argument("--trial-seed", type=int, default=4010)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num-users", type=int, default=10)
    parser.add_argument("--samples-per-client", type=int, default=500)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--local-lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--target-client", type=int, default=1)
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--attack-every", type=int, default=5)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--secure-dp", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--results-root", type=Path, default=default_results_root("membership")
    )
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    torch.set_num_threads(options.torch_threads)
    if (
        min(
            options.rounds,
            options.local_epochs,
            options.local_batch_size,
            options.samples_per_client,
            options.candidate_count,
            options.attack_every,
        )
        <= 0
        or options.num_users < 3
    ):
        raise ValueError("Positive counts and at least three clients are required")
    if not 1 <= options.target_client <= options.num_users:
        raise ValueError("target-client is a node ID from 1 to num-users")
    if options.local_batch_size > options.samples_per_client:
        raise ValueError("local-batch-size must not exceed samples-per-client")
    output = trial_directory(options.results_root, options.dataset, options.trial_seed)
    for scheme in dict.fromkeys(options.schemes):
        if (output / scheme).exists():
            raise FileExistsError(
                f"Refusing to overwrite a scheme trial: {output / scheme}"
            )
    from training.exclusions import exclusion_file, load_exclusions

    config = TrainingConfig(
        exclusion_file=exclusion_file(options.dataset, options.exclude_indices),
        trial_seed=options.trial_seed,
        dataset=options.dataset,
        rounds=options.rounds,
        num_users=options.num_users,
        samples_per_client=options.samples_per_client,
        local_epochs=options.local_epochs,
        local_batch_size=options.local_batch_size,
        local_lr=options.local_lr,
        alpha=options.alpha,
        device=options.device,
        momentum=options.momentum,
        eval_batch_size=options.eval_batch_size,
        data_root=options.data_root,
    )
    seed_everything(options.trial_seed)
    train, test = _load_dataset(config)
    partitions = _sample_partitions(train, config, _build_node_args(config, output))
    initial = _clone_state(_build_model(config).state_dict())
    queries = prepare_queries(
        train,
        partitions,
        options.target_client,
        options.candidate_count,
        options.trial_seed,
        load_exclusions(config.exclusion_file, config.dataset, len(train)),
    )
    output.mkdir(parents=True, exist_ok=True)
    rows, applicability = [], []
    for scheme in dict.fromkeys(options.schemes):
        seed_everything(options.trial_seed)
        rows.extend(
            run_scheme(
                scheme,
                options,
                config,
                train,
                test,
                partitions,
                initial,
                queries,
                output / scheme,
            )
        )
        saved_availability = json.loads(
            (output / scheme / "configuration.json").read_text(encoding="utf-8")
        )
        saved_availability = saved_availability["membership"]
        for attack, details in saved_availability["attacks"].items():
            applicability.append({"scheme": scheme, "attack": attack, **details})
    fingerprints = [
        (
            json.loads((output / s / "configuration.json").read_text(encoding="utf-8"))[
                "membership"
            ]
        )
        for s in dict.fromkeys(options.schemes)
    ]
    if (
        len(
            {
                (v["candidate_fingerprint"], tuple(v["observation_rounds"]))
                for v in fingerprints
            }
        )
        != 1
    ):
        raise RuntimeError(
            "Membership schemes did not use identical candidates and checkpoints"
        )
    if rows:
        write_csv(
            output
            / ("comparison_" + "_".join(dict.fromkeys(options.schemes)) + ".csv"),
            rows,
        )
    write_csv(output / "applicability.csv", applicability)
    print(f"Membership results: {output}")


if __name__ == "__main__":
    main()

"""Reconstruct training inputs from client updates or server aggregates."""

import argparse
from pathlib import Path

import torch

from attacks.gia.inverting import run_attack
from attacks.gia.node_protocol import NodeProtocolConfig, run_node_protocol
from evaluation.paths import default_results_root, trial_directory
from evaluation.reporting import write_csv
from training.setup import (
    _build_model,
    _build_node_args,
    _clone_state,
    _load_dataset,
    _sample_partitions,
    seed_everything,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-indices", default=None)
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["fedavg", "sihefl", "dp_fedavg", "dp_sihefl"],
        default=["fedavg", "sihefl", "dp_fedavg"],
    )
    parser.add_argument("--dataset", choices=["MNIST", "CIFAR10"], default="MNIST")
    for flag, default in [
        ("trial-seed", 4010),
        ("rounds", 20),
        ("attack-round", 1),
        ("num-users", 10),
        ("samples-per-client", 12),
        ("local-epochs", 5),
        ("local-batch-size", 4),
        ("target-client", 2),
        ("iterations", 24000),
        ("restarts", 1),
        ("progress-interval", 100),
        ("eval-batch-size", 128),
        ("torch-threads", 1),
    ]:
        parser.add_argument("--" + flag, type=int, default=default)
    for flag, default in [
        ("alpha", 10.0),
        ("local-lr", 0.01),
        ("momentum", 0.5),
        ("attack-lr", 0.1),
        ("total-variation", 0.01),
        ("noise-multiplier", 1.0),
        ("max-grad-norm", 1.0),
        ("delta", 1e-5),
    ]:
        parser.add_argument("--" + flag, type=float, default=default)
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--secure-dp", action="store_true")
    parser.add_argument(
        "--observations",
        choices=["all", "aggregate", "individual"],
        default="all",
        help="'all' attacks each scheme's actual server-visible observation (FedAvg/DP "
        "individual + everyone's aggregate) so SIHEFL's observation restriction shows; "
        "'aggregate' compares real matched aggregates, allowing CKKS numerical differences",
    )
    parser.add_argument(
        "--simulation",
        choices=["strong", "weak"],
        default="strong",
        help="strong = known local hyperparameters, unknown actual minibatch order; "
        "weak = mismatched non-DP hyperparameters; neither replays private randomness",
    )
    parser.add_argument(
        "--positive-control",
        action="store_true",
        help="Add one synthetic known-schedule FedAvg inversion diagnostic; excluded from primary comparison",
    )
    parser.add_argument(
        "--iterate-selection", choices=["final", "best_objective"], default="final"
    )
    parser.add_argument("--skip-attacks", action="store_true")
    parser.add_argument(
        "--results-root", type=Path, default=default_results_root("gia")
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if (
        min(
            args.rounds,
            args.num_users,
            args.samples_per_client,
            args.local_epochs,
            args.local_batch_size,
            args.iterations,
            args.restarts,
            args.torch_threads,
        )
        < 1
        or args.num_users < 3
    ):
        raise ValueError("Require positive counts and at least three clients")
    if (
        not 1 <= args.attack_round <= args.rounds
        or not 1 <= args.target_client <= args.num_users
    ):
        raise ValueError("Invalid attack round or target client")
    torch.set_num_threads(args.torch_threads)
    fields = NodeProtocolConfig.__dataclass_fields__
    from training.exclusions import exclusion_file

    config = NodeProtocolConfig(
        exclusion_file=exclusion_file(args.dataset, args.exclude_indices),
        **{k: v for k, v in vars(args).items() if k in fields},
    )
    output = trial_directory(args.results_root, args.dataset, args.trial_seed)
    seed_everything(args.trial_seed)
    train, test = _load_dataset(config)
    partitions = _sample_partitions(train, config, _build_node_args(config, output))
    initial = _clone_state(_build_model(config).state_dict())
    schemes = list(dict.fromkeys(args.schemes))
    for scheme in schemes:
        if (output / scheme).exists():
            raise FileExistsError(f"Refusing to overwrite {output / scheme}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scheme in schemes:
        seed_everything(args.trial_seed)
        paths = run_node_protocol(
            scheme, args, config, train, test, partitions, initial, output / scheme
        )
        if not args.skip_attacks:
            for observation in paths["observations"]:
                aggregate = observation["condition"].endswith("server_aggregate")
                if args.observations == "aggregate" and not aggregate:
                    continue
                if args.observations == "individual" and aggregate:
                    continue
                rows.append(
                    run_attack(
                        observation["attack_view"],
                        observation["evaluation_target"],
                        output / scheme / "inversion" / observation["condition"],
                        iterations=args.iterations,
                        restarts=args.restarts,
                        attack_lr=args.attack_lr,
                        total_variation=args.total_variation,
                        device_name=args.device,
                        attack_seed=args.trial_seed,
                        progress_interval=args.progress_interval,
                        iterate_selection=args.iterate_selection,
                    )
                )
    if rows:
        write_csv(
            output / ("comparison_" + "_".join(schemes) + ".csv"),
            [r for r in rows if r["primary_comparison"]],
        )
        controls = [r for r in rows if not r["primary_comparison"]]
        if controls:
            write_csv(output / "positive_control.csv", controls)
    print(f"GIA results: {output}")


if __name__ == "__main__":
    main()

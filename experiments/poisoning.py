"""Evaluate targeted label poisoning against matched clean training."""

import argparse
import json
import sys
from pathlib import Path

import torch

from attacks.poisoning.label_flip import (
    DATASET_MAPPINGS,
    AttackConfig,
    select_malicious_clients,
)
from evaluation.paths import (
    default_results_root,
    poisoning_destination,
    portable_options,
    portable_path,
    trial_directory,
)


def boolean(value):
    if str(value).lower() in ("true", "1", "yes"):
        return True
    if str(value).lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", "--epochs", dest="epochs", type=int, default=20)
    parser.add_argument(
        "--num-users", "--num_users", dest="num_users", type=int, default=10
    )
    parser.add_argument(
        "--samples-per-client",
        "--num_samples",
        dest="num_samples",
        type=int,
        default=500,
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--frac",
        type=float,
        choices=[1.0],
        default=1.0,
        help="Actual nodes currently require full client participation",
    )
    parser.add_argument(
        "--local-epochs", "--local_ep", dest="local_ep", type=int, default=5
    )
    parser.add_argument(
        "--local-batch-size", "--local_bs", dest="local_bs", type=int, default=32
    )
    parser.add_argument("--eval-batch-size", "--bs", dest="bs", type=int, default=128)
    parser.add_argument("--local-lr", "--lr", dest="lr", type=float, default=0.01)
    parser.add_argument(
        "--momentum", type=float, default=0.5, help="Actual SGD momentum (default: 0.5)"
    )
    parser.add_argument("--model", default="cnn")
    parser.add_argument(
        "--partition-size",
        "--partition_size",
        dest="partition_size",
        type=int,
        default=None,
        help="Minimum route size (default: min(10, num-users))",
    )
    parser.add_argument(
        "--output-directory",
        "--output_directory",
        dest="output_directory",
        default="results",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset cache root (default: repository/data)",
    )
    parser.add_argument("--dataset", default="CIFAR10", choices=["MNIST", "CIFAR10"])
    parser.add_argument("--iid", nargs="?", const=True, default=False, type=boolean)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument("--all_clients", default=True, action="store_true")
    parser.add_argument("--exclude-indices", default=None)
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["fedavg", "sihefl", "dp_fedavg", "dp_sihefl"],
        default=["fedavg", "sihefl", "dp_fedavg"],
    )
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--secure-dp", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--results-root", type=Path, default=default_results_root("poisoning")
    )
    parser.add_argument(
        "--clean-results-root",
        type=Path,
        default=None,
        help="Clean controls; defaults to a clean/ sibling of the poisoning root",
    )
    parser.add_argument("--condition", choices=["clean", "attack"], default="attack")
    parser.add_argument(
        "--attack-mapping",
        "--attack_mapping",
        dest="attack_mapping",
        choices=tuple(k for m in DATASET_MAPPINGS.values() for k in m),
        default=None,
    )
    malicious = parser.add_mutually_exclusive_group()
    malicious.add_argument(
        "--malicious-fraction",
        "--malicious_fraction",
        dest="malicious_fraction",
        type=float,
        default=0.2,
    )
    malicious.add_argument("--num-poisoned-clients", type=int, default=None)
    parser.add_argument(
        "--poison-fraction",
        type=float,
        default=1.0,
        help="Fraction of eligible source-class records to flip within each malicious client",
    )
    parser.add_argument("--attack-start-round", type=int, default=1)
    parser.add_argument("--attack-end-round", type=int, default=None)
    parser.add_argument(
        "--trial-seed", "--trial_seed", dest="trial_seed", type=int, default=4010
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run the clean control plus every mapping for this dataset, then summarise",
    )
    parser.add_argument(
        "--skip-clean",
        "--skip_clean",
        dest="skip_clean",
        action="store_true",
        help="Suite: reuse an existing completed clean control instead of re-running it",
    )
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Suite: skip only completed runs whose saved configuration and pairing match",
    )
    parser.add_argument(
        "--output-prefix",
        "--output_prefix",
        dest="output_prefix",
        default="poisoning",
        help="Suite: summary filename prefix",
    )
    args = parser.parse_args(argv)
    if (
        min(args.epochs, args.num_samples, args.local_ep, args.local_bs, args.bs) < 1
        or args.num_users < 3
    ):
        parser.error("Positive training counts and at least three clients are required")
    if args.lr <= 0 or not 0 <= args.momentum < 1 or args.alpha <= 0:
        parser.error("Require positive learning rate/alpha and 0 <= momentum < 1")
    if args.partition_size is None:
        args.partition_size = min(10, args.num_users)
    if not 1 <= args.partition_size <= args.num_users:
        parser.error("partition-size must be between 1 and num-users")
    if args.dataset not in DATASET_MAPPINGS:
        parser.error("Poisoning supports MNIST and CIFAR10")
    if args.resume_completed and not args.suite:
        parser.error("--resume-completed requires --suite")
    args.attack_mapping = args.attack_mapping or next(
        iter(DATASET_MAPPINGS[args.dataset])
    )
    if args.attack_mapping not in DATASET_MAPPINGS[args.dataset]:
        parser.error("Attack mapping does not belong to this dataset")
    if not 0 <= args.poison_fraction <= 1 or not 0 <= args.malicious_fraction <= 1:
        parser.error("Poison and malicious fractions must be between 0 and 1")
    if args.num_poisoned_clients is not None:
        if not 0 <= args.num_poisoned_clients <= args.num_users:
            parser.error("num-poisoned-clients must be between 0 and num-users")
        args.malicious_fraction = args.num_poisoned_clients / args.num_users
    end = args.attack_end_round if args.attack_end_round is not None else args.epochs
    if not 1 <= args.attack_start_round <= end <= args.epochs:
        parser.error("Require 1 <= attack-start-round <= attack-end-round <= rounds")
    return args


def run_scheme(
    scheme,
    options,
    config,
    train,
    test,
    partitions,
    initial,
    attack,
    output,
    recorded_config=None,
):
    from attacks.poisoning.evaluation import calculate_confusion_matrix, mapping_metrics
    from attacks.poisoning.transform import PoisoningTransform
    from evaluation.model import test_fun
    from evaluation.reporting import write_csv
    from training.runner import run_training
    from training.setup import _build_model, _build_node_args

    if attack.mapping.name not in DATASET_MAPPINGS[config.dataset]:
        raise ValueError("Attack mapping does not belong to this dataset")
    transform = PoisoningTransform(attack, partitions)
    transform.prepare(train)

    if attack.enabled and not any(transform.selected.values()):
        raise ValueError(
            f"Enabled attack {attack.mapping.name} flips zero records: no source-class "
            f"({attack.mapping.source_class}) data on malicious clients "
            f"{sorted(attack.malicious_client_ids)}. Check the mapping/dataset, the "
            f"malicious selection, or the non-IID alpha."
        )
    devices = (
        [torch.device(config.device).index or 0]
        if str(config.device).startswith("cuda")
        else []
    )
    with torch.random.fork_rng(devices=devices):
        model = _build_model(config)
    args = _build_node_args(config, output)
    rows = []
    confusion_rows = []

    def observe(round_number, global_state):
        model.load_state_dict(global_state)
        with torch.random.fork_rng(devices=devices):
            accuracy, loss = test_fun(model, test, args)
            confusion = calculate_confusion_matrix(model, test, args)
        for source in range(10):
            for target in range(10):
                confusion_rows.append(
                    dict(
                        round=round_number,
                        source=source,
                        target=target,
                        count=int(confusion[source, target]),
                    )
                )
        for name, mapping in DATASET_MAPPINGS[config.dataset].items():
            rows.append(
                dict(
                    scheme=scheme,
                    round=round_number,
                    condition="attack" if attack.enabled else "clean",
                    selected_attack=name == attack.mapping.name,
                    attack_name=name,
                    accuracy=float(accuracy),
                    test_loss=float(loss),
                    designated_malicious_clients=len(attack.malicious_client_ids),
                    **transform.summary(),
                    **mapping_metrics(confusion, mapping),
                )
            )
        write_csv(output / "poisoning_metrics.csv", rows)
        write_csv(output / "confusion_by_round.csv", confusion_rows)

    run_training(
        scheme,
        options,
        config,
        train,
        test,
        partitions,
        initial,
        output,
        post_round=observe,
        dataset_transform=transform,
        round_start=transform.begin_round,
        experiment_name="poisoning" if attack.enabled else "clean",
        release_rounds=(),
    )
    import hashlib

    digest = hashlib.sha256(json.dumps(partitions, sort_keys=True).encode())
    for name, value in initial.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    payload = dict(recorded_config or {})
    payload.update(
        schema_version=3,
        scheme=scheme,
        condition="attack" if attack.enabled else "clean",
        training=portable_options(config),
        pairing_fingerprint=digest.hexdigest(),
        malicious_client_ids=sorted(attack.malicious_client_ids),
        source_records_by_client=transform.source_counts,
        poisoned_records_by_client={
            str(c): len(ids) for c, ids in transform.selected.items()
        },
        eligible_selected_records=sum(map(len, transform.selected.values())),
        zero_exposure=not any(transform.selected.values()),
    )
    (output / "attack_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return rows


def main(argv=None):
    import hashlib

    from evaluation.poisoning_summary import summarise_directory
    from training.setup import (
        TrainingConfig,
        _build_model,
        _build_node_args,
        _clone_state,
        _load_dataset,
        _sample_partitions,
        seed_everything,
    )

    use_utf8_console()
    args = parse_arguments(argv)
    from training.exclusions import exclusion_file

    reserved = exclusion_file(args.dataset, args.exclude_indices)
    if args.model != "cnn":
        raise ValueError("This experiment supports the repository CNN")
    if Path(args.output_directory).name != args.output_directory:
        raise ValueError("output-directory must be a simple run name")
    torch.set_num_threads(args.torch_threads)
    device = args.device if args.gpu == -1 else f"cuda:{args.gpu}"
    config = TrainingConfig(
        trial_seed=args.trial_seed,
        dataset=args.dataset,
        rounds=args.epochs,
        num_users=args.num_users,
        samples_per_client=args.num_samples,
        alpha=args.alpha,
        local_epochs=args.local_ep,
        local_batch_size=args.local_bs,
        local_lr=args.lr,
        momentum=args.momentum,
        eval_batch_size=args.bs,
        data_root=args.data_root,
        device=device,
        iid=args.iid,
        partition_size=args.partition_size,
        exclusion_file=reserved,
    )
    output = trial_directory(args.results_root, args.dataset, args.trial_seed)
    clean_root = args.clean_results_root or (
        args.results_root.parent / "clean"
        if args.results_root.name == "poisoning"
        else args.results_root / "clean"
    )
    clean_output = trial_directory(clean_root, args.dataset, args.trial_seed)
    if args.output_directory != "results":
        output = output / "variants" / args.output_directory
        clean_output = clean_output / "variants" / args.output_directory
    schemes = list(dict.fromkeys(args.schemes))
    mappings = DATASET_MAPPINGS[args.dataset]

    if args.suite:
        first_mapping = next(iter(mappings))
        conditions = ([] if args.skip_clean else [("clean", first_mapping)]) + [
            ("attack", name) for name in mappings
        ]
    else:
        conditions = [(args.condition, args.attack_mapping)]

    seed_everything(args.trial_seed)
    train, test = _load_dataset(config)
    partitions = _sample_partitions(train, config, _build_node_args(config, output))
    initial = _clone_state(_build_model(config).state_dict())

    digest = hashlib.sha256(json.dumps(partitions, sort_keys=True).encode())
    for name, value in initial.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    fingerprint = digest.hexdigest()
    exclusion_sha = (
        hashlib.sha256(Path(reserved).read_bytes()).hexdigest() if reserved else None
    )
    malicious_ids = select_malicious_clients(
        args.num_users, args.malicious_fraction, args.trial_seed
    )

    # Validate all destinations before starting the suite.
    pending = []
    check_conditions = conditions + (
        [("clean", next(iter(mappings)))] if args.suite and args.skip_clean else []
    )
    for condition, mapping_name in check_conditions:
        attack = AttackConfig(
            condition == "attack",
            mappings[mapping_name],
            malicious_ids,
            args.poison_fraction,
            args.attack_start_round,
            args.attack_end_round,
            args.trial_seed,
        )
        for scheme in schemes:
            destination = poisoning_destination(
                output, scheme, condition, mapping_name, clean_output
            )
            expected = _attack_config_payload(args, condition, attack)
            expected.update(
                pairing_fingerprint=fingerprint,
                exclusion_sha256=exclusion_sha,
                exclusion_file=portable_path(reserved),
                schema_version=3,
                scheme=scheme,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=args.max_grad_norm,
                delta=args.delta,
                secure_dp=args.secure_dp,
            )
            if args.suite and args.skip_clean and condition == "clean":
                _validate_completed_run(destination, expected)
            elif destination.exists():
                if not args.resume_completed:
                    raise FileExistsError(
                        f"Refusing to overwrite {destination}; use --suite --resume-completed for a matching completed run"
                    )
                _validate_completed_run(destination, expected)
                print(f"Skipping completed poisoning run: {destination}")
            else:
                pending.append((condition, mapping_name, scheme, attack, expected))

    for condition, mapping_name, scheme, attack, expected in pending:
        seed_everything(args.trial_seed)
        destination = poisoning_destination(
            output, scheme, condition, mapping_name, clean_output
        )
        run_scheme(
            scheme,
            args,
            config,
            train,
            test,
            partitions,
            initial,
            attack,
            destination,
            recorded_config=expected,
        )
        observed_fingerprint = json.loads(
            (destination / "attack_config.json").read_text(encoding="utf-8")
        )["pairing_fingerprint"]
        if observed_fingerprint != fingerprint:
            raise RuntimeError(
                f"Pairing fingerprint changed during training: {destination}"
            )
    if args.suite:
        summarise_directory(
            output,
            output / f"{args.output_prefix}_summary.csv",
            clean_directory=clean_output,
        )
    print(f"Poisoning results: {output}")


def _attack_config_payload(args, condition, attack_config):
    mapping = attack_config.mapping
    payload = {
        "condition": condition,
        "trial_seed": args.trial_seed,
        "dataset": args.dataset,
        "attack_name": mapping.name,
        "source_class": mapping.source_class,
        "source_name": mapping.source_name,
        "target_class": mapping.target_class,
        "target_name": mapping.target_name,
        "malicious_fraction": args.malicious_fraction,
        "malicious_client_ids": sorted(attack_config.malicious_client_ids),
        "num_users": args.num_users,
        "samples_per_client": args.num_samples,
        "global_rounds": args.epochs,
        "local_epochs": args.local_ep,
        "local_batch_size": args.local_bs,
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "eval_batch_size": args.bs,
        "data_root": portable_path(args.data_root),
        "iid": args.iid,
        "partition_size": args.partition_size,
        "poison_fraction": attack_config.poison_fraction,
        "attack_start_round": attack_config.start_round,
        "attack_end_round": attack_config.end_round,
        "dirichlet_alpha": args.alpha,
        "execution": "shared runner with actual selected scheme nodes",
        "serialized_payload_measurement": "cost.csv (uniform across schemes)",
    }
    return payload


def _validate_completed_run(destination, expected):
    required = (
        "status.json",
        "attack_config.json",
        "poisoning_metrics.csv",
        "utility.csv",
        "cost.csv",
    )
    if not destination.is_dir() or any(
        not (destination / name).is_file() for name in required
    ):
        raise FileNotFoundError(
            f"Cannot resume incomplete poisoning run: {destination}"
        )
    try:
        status = json.loads((destination / "status.json").read_text(encoding="utf-8"))
        saved = json.loads(
            (destination / "attack_config.json").read_text(encoding="utf-8")
        )
        if saved.get("schema_version", 2) < 3:
            exposure = json.loads(
                (destination / "exposure.json").read_text(encoding="utf-8")
            )
        else:
            exposure = saved
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"Cannot resume unreadable poisoning run: {destination}"
        ) from exc
    if status.get("training") != "complete":
        raise ValueError(f"Cannot resume unfinished poisoning run: {destination}")

    differences = [
        key
        for key, value in expected.items()
        if key not in ("exclusion_file", "schema_version") and saved.get(key) != value
    ]
    if (
        differences
        or exposure.get("pairing_fingerprint") != expected["pairing_fingerprint"]
    ):
        raise ValueError(
            f"Existing poisoning run differs from requested study at {destination}: {differences or ['exposure fingerprint']}"
        )


def use_utf8_console():
    """Allow the base experiment summary to print symbols on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()

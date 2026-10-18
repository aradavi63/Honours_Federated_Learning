"""Train a federated model and record utility, privacy and communication costs."""

import argparse
from pathlib import Path

import torch

from evaluation.paths import default_results_root, trial_directory
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-indices", default=None)
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
        "--scheme",
        choices=["fedavg", "sihefl", "dp_fedavg", "dp_sihefl"],
        default="fedavg",
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
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--secure-dp", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--results-root", type=Path, default=default_results_root("training")
    )
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    if (
        min(
            options.rounds,
            options.local_epochs,
            options.local_batch_size,
            options.samples_per_client,
            options.torch_threads,
        )
        <= 0
        or options.num_users < 3
    ):
        raise ValueError("Positive counts and at least three clients are required")
    if options.local_batch_size > options.samples_per_client:
        raise ValueError("local-batch-size must not exceed samples-per-client")
    torch.set_num_threads(options.torch_threads)
    from training.exclusions import exclusion_file

    config = TrainingConfig(
        exclusion_file=exclusion_file(options.dataset, options.exclude_indices),
        **{
            name: getattr(options, name)
            for name in (
                "trial_seed",
                "dataset",
                "rounds",
                "num_users",
                "samples_per_client",
                "alpha",
                "local_epochs",
                "local_batch_size",
                "local_lr",
                "device",
                "momentum",
                "eval_batch_size",
                "data_root",
            )
        },
    )
    output = (
        trial_directory(options.results_root, options.dataset, options.trial_seed)
        / options.scheme
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    seed_everything(options.trial_seed)
    train, test = _load_dataset(config)
    partitions = _sample_partitions(train, config, _build_node_args(config, output))
    initial = _clone_state(_build_model(config).state_dict())
    model = run_training(
        options.scheme,
        options,
        config,
        train,
        test,
        partitions,
        initial,
        output,
        release_rounds=(),
    )
    torch.save(_clone_state(model.state_dict()), output / "global_model.pt")
    print(f"Training results: {output}")


if __name__ == "__main__":
    main()

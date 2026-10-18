"""Create paired clean-versus-attack recall and ASR differences."""

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "scheme",
    "trial_seed",
    "round",
    "attack_name",
    "source_class",
    "target_class",
    "clean_accuracy",
    "attack_accuracy",
    "accuracy_change_pp",
    "clean_source_recall",
    "attack_source_recall",
    "source_recall_loss_pp",
    "clean_source_to_target_asr",
    "attack_source_to_target_asr",
    "asr_increase_pp",
    "clean_other_class_accuracy",
    "attack_other_class_accuracy",
    "other_class_accuracy_change_pp",
    "active_poisoning_clients",
    "flipped_source_records",
    "eligible_selected_records",
    "zero_exposure",
    "target_recall_change_pp",
    "remaining_class_recall_change_pp",
]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-directory", "--results_directory", type=Path, required=True
    )
    parser.add_argument(
        "--clean-directory",
        type=Path,
        help="clean controls for a separate poisoning trial",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-prefix", default=None)
    args = parser.parse_args(argv)

    clean_directory = args.clean_directory
    trial = args.results_directory
    if clean_directory is None and trial.parent.parent.name == "poisoning":
        sibling = trial.parent.parent.parent / "clean" / trial.parent.name / trial.name
        if sibling.is_dir():
            clean_directory = sibling
    summarise_directory(
        args.results_directory,
        args.output or args.results_directory / "poisoning_summary.csv",
        args.run_prefix,
        clean_directory=clean_directory,
    )


def summarise_directory(
    result_directory, output, run_prefix=None, clean_directory=None
):
    """Validate matched controls and write paired poisoning effects."""
    result_directory = Path(result_directory)
    runs = load_runs(result_directory, run_prefix=run_prefix)
    if clean_directory is not None:
        runs += load_runs(Path(clean_directory), run_prefix=run_prefix)
    if not runs:
        raise ValueError(f"No poisoning runs found under {result_directory}")

    clean_by_seed = {}
    for config, rows in runs:
        if config["condition"] == "clean":
            key = pairing_key(config)
            if key in clean_by_seed:
                raise ValueError(
                    "Ambiguous clean controls; select a narrower results directory"
                )
            clean_by_seed[key] = (config, rows)

    summary = []
    for attack_config, attack_rows in runs:
        if attack_config["condition"] != "attack":
            continue
        seed = attack_config["trial_seed"]
        pair = pairing_key(attack_config)
        if pair not in clean_by_seed:
            raise ValueError(f"No paired clean run found for trial seed {seed}")
        _, clean_rows = clean_by_seed[pair]
        clean_lookup = {
            (row["attack_name"], int(row["round"])): row for row in clean_rows
        }
        for attack_row in attack_rows:
            if attack_row["selected_attack"].lower() != "true":
                continue
            key = (attack_row["attack_name"], int(attack_row["round"]))
            if key not in clean_lookup:
                raise ValueError(
                    f"Paired clean run lacks {key[0]} round {key[1]} for "
                    f"scheme {attack_config.get('scheme', 'sihefl')} seed {seed}"
                )
            clean_row = clean_lookup[key]
            row = compare_rows(seed, clean_row, attack_row)
            row["scheme"] = attack_config.get("scheme", "sihefl")
            summary.append(row)

    if not summary:
        raise ValueError("No attacked rows were available for paired comparison")
    summary.sort(
        key=lambda row: (
            row["trial_seed"],
            row["scheme"],
            row["attack_name"],
            row["round"],
        )
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    for row in summary:
        print(
            f"seed={row['trial_seed']} round={row['round']} "
            f"attack={row['attack_name']} "
            f"recall_loss={row['source_recall_loss_pp']:.2f} pp "
            f"ASR_increase={row['asr_increase_pp']:.2f} pp"
        )
    print(f"Wrote {output}")
    return summary


def pairing_key(config):
    # Do not pair a DP attack with plaintext utility or a differently trained model.
    fields = (
        "trial_seed",
        "dataset",
        "num_users",
        "global_rounds",
        "local_epochs",
        "local_batch_size",
        "learning_rate",
        "momentum",
        "dirichlet_alpha",
        "samples_per_client",
        "iid",
        "partition_size",
        "data_root",
    )
    values = tuple(config.get(k) for k in fields)
    scheme = config.get("scheme", "sihefl")
    dp = (
        tuple(
            config.get(k)
            for k in ("noise_multiplier", "max_grad_norm", "delta", "secure_dp")
        )
        if scheme in ("dp_fedavg", "dp_sihefl")
        else ()
    )
    return (
        scheme,
        *values,
        *dp,
        config.get("pairing_fingerprint"),
        config.get("exclusion_sha256"),
        tuple(config.get("malicious_client_ids", [])),
    )


def load_runs(result_directory, run_prefix=None):
    runs = []
    for config_path in sorted(result_directory.rglob("attack_config.json")):
        relative = config_path.relative_to(result_directory)
        if run_prefix is not None and (
            not relative.parts or not relative.parts[0].startswith(run_prefix + "_")
        ):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics_path = config_path.with_name("poisoning_metrics.csv")
        if not metrics_path.exists():
            continue
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            runs.append((config, list(csv.DictReader(handle))))
    return runs


def compare_rows(seed, clean, attack):
    clean_accuracy = float(clean["accuracy"])
    attack_accuracy = float(attack["accuracy"])
    clean_recall = float(clean["source_recall"])
    attack_recall = float(attack["source_recall"])
    clean_asr = float(clean["source_to_target_asr"])
    attack_asr = float(attack["source_to_target_asr"])
    clean_other = float(clean["other_class_accuracy"])
    attack_other = float(attack["other_class_accuracy"])
    return {
        "target_recall_change_pp": float(attack["target_recall"])
        - float(clean["target_recall"])
        if "target_recall" in attack and "target_recall" in clean
        else "",
        "remaining_class_recall_change_pp": float(
            attack["remaining_class_macro_recall"]
        )
        - float(clean["remaining_class_macro_recall"])
        if "remaining_class_macro_recall" in attack
        and "remaining_class_macro_recall" in clean
        else "",
        "eligible_selected_records": attack.get("eligible_selected_records", ""),
        "zero_exposure": attack.get("zero_exposure", ""),
        "trial_seed": seed,
        "round": int(attack["round"]),
        "attack_name": attack["attack_name"],
        "source_class": int(attack["source_class"]),
        "target_class": int(attack["target_class"]),
        "clean_accuracy": clean_accuracy,
        "attack_accuracy": attack_accuracy,
        "accuracy_change_pp": attack_accuracy - clean_accuracy,
        "clean_source_recall": clean_recall,
        "attack_source_recall": attack_recall,
        "source_recall_loss_pp": clean_recall - attack_recall,
        "clean_source_to_target_asr": clean_asr,
        "attack_source_to_target_asr": attack_asr,
        "asr_increase_pp": attack_asr - clean_asr,
        "clean_other_class_accuracy": clean_other,
        "attack_other_class_accuracy": attack_other,
        "other_class_accuracy_change_pp": attack_other - clean_other,
        "active_poisoning_clients": int(attack["active_poisoning_clients"]),
        "flipped_source_records": int(attack["flipped_source_records"]),
    }


if __name__ == "__main__":
    main()

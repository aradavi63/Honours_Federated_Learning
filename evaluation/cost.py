"""Record per-round wall time, serialized payload sizes and transmission counts.

Plaintext payloads use serialized model states; encrypted payloads use CKKS
ciphertexts. Observer time is recorded separately from protocol time."""

import csv
import io
import threading
from pathlib import Path
from statistics import mean

import torch

from schemes import ENCRYPTED_SCHEMES


def serialized_state_bytes(state):
    """Serialized byte size of one model state_dict (the plaintext client payload)."""
    buffer = io.BytesIO()
    torch.save({name: value.detach().cpu() for name, value in state.items()}, buffer)
    return int(buffer.getbuffer().nbytes)


def _serialized_ciphertext_bytes(encrypted_weights):
    return sum(
        len(chunk.serialize())
        for chunks in encrypted_weights.values()
        for chunk in chunks
    )


class CipherByteRecorder:
    """Serialized CKKS bytes per round, by wrapping the native encrypt/decrypt."""

    def __init__(self):
        self._lock = threading.Lock()
        self.client_upload = {}  # round -> summed per-client ciphertext bytes
        self.aggregate = {}  # round -> aggregate ciphertext bytes at the server

    def instrument(self, network, server):
        original_encrypt = network.encryptWeights

        def measured_encrypt(*args, **kwargs):
            encrypted, plaintext = original_encrypt(*args, **kwargs)
            round_number = int(network.overhead_info["epoch_num"]) + 1
            size = _serialized_ciphertext_bytes(encrypted)
            with self._lock:
                self.client_upload[round_number] = (
                    self.client_upload.get(round_number, 0) + size
                )
            return encrypted, plaintext

        network.encryptWeights = measured_encrypt

        original_decrypt = server.decryptWeights

        def measured_decrypt(encrypted_weights, *args, **kwargs):
            round_number = int(server.overhead_info["epoch_num"]) + 1
            size = _serialized_ciphertext_bytes(encrypted_weights)
            with self._lock:
                self.aggregate[round_number] = size
            return original_decrypt(encrypted_weights, *args, **kwargs)

        server.decryptWeights = measured_decrypt


FIELDS = [
    "round",
    "round_seconds",
    "instrumented_round_seconds",
    "observer_seconds",
    "client_upload_bytes_total",
    "aggregate_bytes",
    "transmissions",
]

PROCEDURES = (
    "training_times",
    "noise_calc_time",
    "key_generation_time",
    "encryption_times",
    "decryption_times",
    "aggregation_times",
    "update_times",
)
HE_PROCEDURES = {
    "noise_calc_time",
    "key_generation_time",
    "encryption_times",
    "decryption_times",
}


def write_procedure_times(output, scheme, overhead):
    """Save native procedure timings; client timings are means across clients."""
    rounds = len(overhead["epoch_times"])
    columns = {}
    for name in PROCEDURES:
        if scheme not in ENCRYPTED_SCHEMES and name in HE_PROCEDURES:
            columns[name] = [""] * rounds
        else:
            values = overhead.get(name, [])
            if len(values) != rounds:
                raise ValueError(f"Incomplete procedure timing series: {name}")
            columns[name] = values
    with (Path(output) / "procedure_times.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", *PROCEDURES])
        writer.writeheader()
        writer.writerows(
            {
                "round": index + 1,
                **{name: values[index] for name, values in columns.items()},
            }
            for index in range(rounds)
        )


def _per_round_transmissions(overhead, is_he, rounds):
    if is_he:
        other = overhead.get("other_num_transmissions", [])
        noise = overhead.get("noise_calc_num_transmissions", [])
        return [
            int(
                (other[i] if i < len(other) else 0)
                + (noise[i] if i < len(noise) else 0)
            )
            for i in range(rounds)
        ]
    counts = overhead.get("num_transmissions", [])
    return [int(counts[i]) if i < len(counts) else 0 for i in range(rounds)]


def summarise_cost(
    scheme, overhead, num_clients, model_state, cipher=None, instrumentation_times=None
):
    """Return (per-round rows, summary dict) for one completed run."""
    is_he = scheme in ENCRYPTED_SCHEMES
    epoch_times = [float(t) for t in overhead.get("epoch_times", [])]
    rounds = len(epoch_times)
    transmissions = _per_round_transmissions(overhead, is_he, rounds)
    if is_he:
        upload = [
            int(cipher.client_upload.get(i + 1, 0)) if cipher else 0
            for i in range(rounds)
        ]
        aggregate = [
            int(cipher.aggregate.get(i + 1, 0)) if cipher else 0 for i in range(rounds)
        ]
    else:
        per_client = serialized_state_bytes(model_state)
        upload = [per_client * int(num_clients)] * rounds
        aggregate = [per_client] * rounds
    observer = {r["round"]: r["observer_seconds"] for r in instrumentation_times or []}
    protocol_times = [
        max(0.0, t - observer.get(i + 1, 0.0)) for i, t in enumerate(epoch_times)
    ]
    rows = [
        {
            "round": i + 1,
            "round_seconds": round(protocol_times[i], 4),
            "instrumented_round_seconds": round(epoch_times[i], 4),
            "observer_seconds": round(observer.get(i + 1, 0.0), 4),
            "client_upload_bytes_total": upload[i],
            "aggregate_bytes": aggregate[i],
            "transmissions": transmissions[i],
        }
        for i in range(rounds)
    ]
    summary = {
        "scheme": scheme,
        "rounds": rounds,
        "total_seconds": round(sum(protocol_times), 4),
        "mean_round_seconds": round(mean(protocol_times), 4) if protocol_times else 0.0,
        "timing_boundary": "Native round wall time minus synchronous release export and attack/evaluator observer; includes utility evaluation, boundary copies, ciphertext byte instrumentation and simulator scheduling",
        "mean_client_upload_bytes_per_round": round(mean(upload), 1) if upload else 0.0,
        "mean_aggregate_bytes_per_round": round(mean(aggregate), 1)
        if aggregate
        else 0.0,
        "mean_transmissions_per_round": round(mean(transmissions), 2)
        if transmissions
        else 0.0,
        "communication_boundary": (
            "serialized transmitted payload per round: state_dict "
            "for FedAvg/DP-FedAvg, CKKS ciphertext for SIHEFL/DP-SIHEFL"
        ),
    }
    return rows, summary


def write_cost(
    output,
    scheme,
    overhead,
    num_clients,
    model_state,
    cipher=None,
    instrumentation_times=None,
):
    """Write the per-round cost measurements."""
    rows, summary = summarise_cost(
        scheme, overhead, num_clients, model_state, cipher, instrumentation_times
    )
    output = Path(output)
    with (output / "cost.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_procedure_times(output, scheme, overhead)
    return summary

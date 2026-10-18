"""Run federated training, record utility and costs, and capture server releases."""

import copy
import json
import math
import threading
import time
from pathlib import Path

import torch

from evaluation.experiment import write_configuration
from evaluation.model import test_fun
from evaluation.paths import portable_text
from evaluation.reporting import write_csv
from schemes import ENCRYPTED_SCHEMES, PRIVATE_SCHEMES, SCHEMES
from schemes import fedavg as standard
from schemes.sihefl import server_node as he_server
from schemes.sihefl.client_node import ClientNodeClass as HEClient
from schemes.sihefl.network_node import NetworkSimulationClass as HENetwork
from training.releases import capture_release
from training.setup import (
    DummyAxis,
    DummyCanvas,
    _build_model,
    _build_node_args,
    _clone_state,
)


class QuietNetwork:
    """Headless callbacks and worker-failure propagation for scheme networks."""

    def __init__(self, args):
        super().__init__(args)
        self.failure = None
        self.failure_lock = threading.Lock()

    def record_failure(self, error):
        with self.failure_lock:
            if self.failure is None:
                self.failure = error

    def raise_if_failed(self):
        if self.failure is not None:
            raise RuntimeError(
                "Client/protocol worker failed; aborting round"
            ) from self.failure

    def messageSingleNode(self, *args, **kwargs):
        try:
            return super().messageSingleNode(*args, **kwargs)
        except Exception as error:
            self.record_failure(error)

    def updateText(self, message, widget):
        if (
            message.startswith("+++")
            or message.startswith("Epoch ")
            or "NaN" in message
        ):
            print(
                message.replace("Epoch ", "Round ").replace("Train Acc:", "Test Acc:"),
                flush=True,
            )

    def updateDisplayNetwork(self, *args):
        pass

    def updatePlots(self, *args):
        pass


class PlainNetwork(QuietNetwork, standard.NetworkSimulationClass):
    pass


class EncryptedNetwork(QuietNetwork, HENetwork):
    pass


class ServerCapture:
    """Capture round-start parameters before clients begin local training."""

    def updateOverheadDict(self, epoch_num):
        super().updateOverheadDict(epoch_num)
        self.initial = _clone_state(self.global_model.state_dict())
        self.observed_clients = {}
        self.observed_aggregate = None
        if getattr(self, "round_start", None):
            self.round_start(epoch_num + 1)


class PlainServer(ServerCapture, standard.ServerNodeClass):
    def receiveMessage(self, sender_id, message):
        if len(message) == 1 and "RESULTS" in message and sender_id in self.node_list:
            self.observed_clients[sender_id] = _clone_state(message["RESULTS"][0])
        # Copy the update before the server records its receipt.
        super().receiveMessage(sender_id, message)


class EncryptedServer(ServerCapture, he_server.ServerNodeClass):
    def decryptWeights(self, *args, **kwargs):
        weights, size = super().decryptWeights(*args, **kwargs)
        self.observed_aggregate = _clone_state(weights)
        return weights, size


class FailureAwareClient:
    def receiveMessage(self, *args, **kwargs):
        try:
            return super().receiveMessage(*args, **kwargs)
        except Exception as error:
            self.network.record_failure(error)

    def client_training(self, *args, **kwargs):
        try:
            return super().client_training(*args, **kwargs)
        except Exception as error:
            self.network.record_failure(error)


class PlainClient(FailureAwareClient, standard.ClientNodeClass):
    pass


class EncryptedClient(FailureAwareClient, HEClient):
    pass


def run_training(
    scheme,
    options,
    config,
    train,
    test,
    partitions,
    initial,
    output,
    observer=None,
    dataset_transform=None,
    round_start=None,
    instrument=None,
    experiment_name="training",
    release_rounds=None,
    post_round=None,
):
    """Train the selected scheme and record results.

    release_rounds selects observed rounds; None observes every round.
    Observers receive server-visible models. post_round receives the aggregate.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme: {scheme}")
    selected_releases = None if release_rounds is None else frozenset(release_rounds)
    if selected_releases is not None and any(
        not isinstance(r, int) or not 1 <= r <= config.rounds for r in selected_releases
    ):
        raise ValueError("release_rounds must contain valid one-based round numbers")
    output = Path(output).resolve()
    args = _build_node_args(config, output / "protocol")
    if set(partitions) != set(range(1, config.num_users + 1)):
        raise ValueError("Partitions must contain each configured client exactly once")
    indices = [int(i) for values in partitions.values() for i in values]
    if len(indices) != len(set(indices)) or any(
        i < 0 or i >= len(train) for i in indices
    ):
        raise ValueError(
            "Client partitions must be disjoint and inside the training dataset"
        )
    if any(len(values) != config.samples_per_client for values in partitions.values()):
        raise ValueError(
            "Client sample counts differ from the configured equal-size training"
        )
    if scheme in PRIVATE_SCHEMES:
        if (
            not all(
                math.isfinite(v)
                for v in (
                    options.noise_multiplier,
                    options.max_grad_norm,
                    options.delta,
                )
            )
            or options.noise_multiplier <= 0
            or options.max_grad_norm <= 0
            or not 0 < options.delta < 1
        ):
            raise ValueError("Invalid DP noise, clipping norm or delta")
    output.mkdir(parents=True, exist_ok=False)
    poisoning_run = experiment_name in ("clean", "poisoning")
    if not poisoning_run:
        write_configuration(output, experiment_name, scheme, config, options, len(test))

    def record_status(payload):
        if poisoning_run:
            (output / "status.json").write_text(json.dumps(payload), encoding="utf-8")

    is_he = scheme in ENCRYPTED_SCHEMES
    network = EncryptedNetwork(args) if is_he else PlainNetwork(args)
    server = (
        EncryptedServer(0, network, args) if is_he else PlainServer(0, network, args)
    )
    server.round_start = round_start
    from evaluation.cost import CipherByteRecorder, write_cost

    cipher_recorder = CipherByteRecorder() if is_he else None
    if cipher_recorder is not None:
        cipher_recorder.instrument(network, server)
    if instrument:
        instrument(network, server)
    network.addNode(server)
    trainer, report_privacy = None, None
    if scheme in PRIVATE_SCHEMES:
        if scheme == "dp_sihefl":
            from schemes.dp_sihefl import make_private_trainer
        else:
            from schemes.dp_fedavg import make_private_trainer
        trainer, report_privacy = make_private_trainer(
            partitions,
            options.noise_multiplier,
            options.max_grad_norm,
            options.delta,
            options.secure_dp,
        )
    for client in partitions:
        cls = EncryptedClient if is_he else PlainClient
        network.addNode(
            cls(
                client,
                network,
                args,
                trainer_factory=trainer,
                dataset_transform=dataset_transform,
            )
        )
    server.getNodeList(network.getNodes())
    model = _build_model(config)
    model.load_state_dict(initial)
    server.global_model = model
    utility, privacy, instrumentation_times = [], [], []

    def save_completed_rounds():
        # Keep completed rounds if a later client or attack observer fails.
        temporary = output / "utility.csv.tmp"
        write_csv(temporary, utility)
        temporary.replace(output / "utility.csv")
        if privacy:
            temporary = output / "privacy.json.tmp"
            temporary.write_text(
                json.dumps(
                    {
                        "unit": "record",
                        "adjacency": "add_remove_with_fixed_client_assignment",
                        "accountant": "RDP",
                        "composition": "all local steps across all rounds per client",
                        "noise_multiplier": options.noise_multiplier,
                        "max_grad_norm": options.max_grad_norm,
                        "secure_mode": options.secure_dp,
                        "raw_training_loss_released": False,
                        "rounds": privacy,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            temporary.replace(output / "privacy.json")

    def evaluate_round(global_model, dataset, evaluation_args):
        test_args = copy.copy(evaluation_args)
        test_args.local_bs = evaluation_args.bs
        accuracy, loss = test_fun(global_model, dataset, test_args)
        round_number = int(network.overhead_info["epoch_num"]) + 1
        utility.append(
            {
                "round": round_number,
                "test_accuracy": float(accuracy),
                "test_loss": float(loss),
            }
        )
        if report_privacy:
            privacy.append({"round": round_number, "clients": report_privacy()})
        if not is_he and set(server.observed_clients) != set(partitions):
            raise RuntimeError("The server did not receive every client update")
        save_completed_rounds()
        capture_started = time.perf_counter()
        aggregate = server.observed_aggregate if is_he else global_model.state_dict()
        if aggregate is None:
            raise RuntimeError("No decrypted and unmasked aggregate was captured")
        for name, value in global_model.state_dict().items():
            if not torch.equal(value.detach().cpu(), aggregate[name].detach().cpu()):
                raise RuntimeError(
                    "Captured aggregate differs from the evaluated global model"
                )
        if selected_releases is None or round_number in selected_releases:
            evidence = capture_release(
                scheme,
                round_number,
                server.initial,
                server.observed_clients,
                aggregate,
                partitions,
                routes=server.route if is_he else None,
            )
            if observer:
                # Callbacks receive copies of the server-visible models.
                observer(
                    round_number,
                    evidence["round_start"],
                    evidence["individual"],
                    evidence["aggregate"],
                )
        if post_round:
            post_round(round_number, _clone_state(aggregate))
        instrumentation_times.append(
            dict(
                round=round_number,
                observer_seconds=time.perf_counter() - capture_started,
            )
        )
        return accuracy, loss

    server.evaluator = evaluate_round
    record_status({"training": "running"})
    try:
        server.trainingProcess(
            model,
            train,
            test,
            partitions,
            None,
            None,
            None,
            DummyCanvas(),
            DummyAxis(),
            network.overhead_info,
            DummyAxis(),
            DummyAxis(),
            DummyCanvas(),
        )
        network.raise_if_failed()
    except Exception as error:
        record_status({"training": "failed", "error": portable_text(error)})
        raise
    save_completed_rounds()
    try:
        write_cost(
            output,
            scheme,
            network.overhead_info,
            config.num_users,
            _clone_state(model.state_dict()),
            cipher_recorder,
            instrumentation_times=instrumentation_times,
        )
    except Exception as error:
        record_status({"training": "failed", "error": portable_text(error)})
        raise
    record_status({"training": "complete", "completed_rounds": len(utility)})
    return model

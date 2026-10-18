"""SIHEFL client training, masking and encrypted route messages."""

import copy
import secrets
import threading
import time

from training.local import LocalUpdateCNN

_PRIVATE_RNG = secrets.SystemRandom()


class ClientNodeClass:
    def __init__(
        self, node_id, network, args, trainer_factory=None, dataset_transform=None
    ):
        super().__init__()
        self.trainer_factory = trainer_factory or LocalUpdateCNN
        self.dataset_transform = dataset_transform
        self.node_id = node_id
        self.network = network
        self.args = args
        self.node_list = None
        self.received_encrypted_weights = None
        self.route = None
        self.parition_numbers = []

    def receiveMessage(self, sender_id, message):
        if len(message.keys()) != 1:
            return

        if (
            "ENCRYPTED_WEIGHTS" in message.keys()
            and sender_id in self.node_list
            and sender_id == self.predecessor_id
        ):
            self.received_encrypted_weights = message["ENCRYPTED_WEIGHTS"][0]
            self.local_losses = message["ENCRYPTED_WEIGHTS"][1]

        if "CALC_NOISE" in message.keys() and sender_id == 0:
            self.noiseCalculation()

        if (
            "NOISE_PARTITION" in message.keys()
            and sender_id in self.node_list
            and sender_id in self.route
        ):
            self.parition_numbers.append(message["NOISE_PARTITION"])

    def noise_generation(self):
        noise_partitions = []
        num_of_participates = len(self.route)
        for i in range(num_of_participates):
            noise_partitions.append(round(_PRIVATE_RNG.uniform(10, 50), 4))
        return noise_partitions

    def noiseCalculation(self):

        threads = []
        for index, node in enumerate(self.route):
            if self.node_id != node:
                t = threading.Thread(
                    target=self.network.messageSingleNode,
                    args=(
                        self.node_id,
                        node,
                        {"NOISE_PARTITION": self.noise_partitions[index]},
                        "noise",
                    ),
                )
                t.start()
                threads.append(t)
            else:
                self.parition_numbers.append(self.noise_partitions[index])

        for t in threads:
            t.join()

        while len(self.parition_numbers) != len(self.route):
            getattr(self.network, "raise_if_failed", lambda: None)()
            time.sleep(0.01)

        parition_sum = sum(self.parition_numbers)

        self.network.messageCentralServer(
            self.node_id, {"FINAL_NOISE_VALUE": parition_sum}, "noise"
        )

        self.parition_numbers = []
        self.noise_partitions = []

    def client_training(
        self,
        route,
        client_id,
        dataset_train,
        dict_party_user,
        net_glob,
        stoi,
        tokenizer,
        text_widget,
        context,
        overhead_info,
        train_time_list,
        encryption_time_list,
        aggregate_time_list,
        weight_size_noise_list,
        G,
        visualisation_canvas,
        visualisation_ax,
        colours,
        pos,
    ):
        self.route = []
        self.node_list = []
        for partition in route:
            if self.node_id in partition:
                self.route = partition
            self.node_list += partition
        p = self.route.index(self.node_id)
        if p == 0:
            self.predecessor_id = None
        else:
            self.predecessor_id = self.route[p - 1]
        if p == len(self.route) - 1:
            self.successor_id = None
        else:
            self.successor_id = self.route[p + 1]

        self.network.updateText(f"Starting training on client {client_id}", text_widget)
        training_data = (
            self.dataset_transform(client_id, dataset_train)
            if self.dataset_transform
            else dataset_train
        )
        local = self.trainer_factory(
            args=self.args, dataset=training_data, idxs=dict_party_user[client_id]
        )

        net_glob.load_state_dict(copy.deepcopy(net_glob.state_dict()))
        # The key, route and model distribution count as one transmission.
        overhead_info["other_num_transmissions"][overhead_info["epoch_num"]] += 1

        startTrainTime = time.time()
        local_weights, loss = local.train(
            net=copy.deepcopy(net_glob).to(self.args.device)
        )
        train_time_list.append(time.time() - startTrainTime)

        if self.network.checkForNan(
            local_weights,
            f"Client {client_id} local weights after training",
            text_widget,
        ):
            raise ValueError(
                f"NaN detected in Client {client_id} local weights after training."
            )

        clients_index = list(G.nodes()).index(client_id)
        colours[clients_index] = "orange"
        self.network.updateDisplayNetwork(
            G, visualisation_canvas, visualisation_ax, colours, pos
        )

        self.network.updateText(
            f"Client {client_id} has completed training. Loss: {loss:.4f}", text_widget
        )

        if self.network.checkForNan(
            local_weights,
            f"Client {client_id} local weights before encryption",
            text_widget,
        ):
            raise ValueError(
                f"NaN detected in Client {client_id} local weights before encryption."
            )

        self.noise_partitions = self.noise_generation()
        self.noise = round(sum(self.noise_partitions), 4)

        startEncryptTime = time.time()
        encrypted_weights, weight_size_noise = self.network.encryptWeights(
            local_weights, context, text_widget, self.noise
        )
        encryption_time_list.append(time.time() - startEncryptTime)
        weight_size_noise_list.append(weight_size_noise)

        while (
            self.received_encrypted_weights is None and self.predecessor_id is not None
        ):
            getattr(self.network, "raise_if_failed", lambda: None)()
            time.sleep(0.1)

        startAggregateTime = time.time()
        if self.predecessor_id is not None:
            current_encrypted_weights = self.network.aggregateEncryptedWeights(
                self.received_encrypted_weights, encrypted_weights, text_widget
            )
        else:
            current_encrypted_weights = encrypted_weights
            self.local_losses = []
        aggregate_time_list.append(time.time() - startAggregateTime)

        self.local_losses.append(loss)

        if self.successor_id is not None:
            self.network.messageSingleNode(
                self.node_id,
                self.successor_id,
                {
                    "ENCRYPTED_WEIGHTS": [
                        current_encrypted_weights,
                        self.local_losses.copy(),
                    ]
                },
                "other",
            )
        else:
            self.network.messageCentralServer(
                self.node_id,
                {
                    "ENCRYPTED_WEIGHTS": [
                        current_encrypted_weights,
                        self.local_losses.copy(),
                    ]
                },
                "other",
            )

        self.local_losses = []
        self.received_encrypted_weights = None

        colours[clients_index] = "green"
        self.network.updateDisplayNetwork(
            G, visualisation_canvas, visualisation_ax, colours, pos
        )

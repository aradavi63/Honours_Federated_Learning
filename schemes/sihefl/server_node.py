"""SIHEFL round coordination, mask removal and model aggregation."""

import math
import random
import statistics
import threading
import time

import networkx as nx
import numpy as np
import tenseal as ts
import torch

from evaluation.model import test_fun


class ServerNodeClass:
    def __init__(self, node_id, network, args, evaluator=None):
        super().__init__()
        self.evaluator = evaluator or test_fun
        self.node_id = node_id
        self.network = network
        self.args = args
        self.node_list = []
        self.route = []
        self.predecessors = []
        self.received_encrypted_weights_list = []
        self.received_encrypted_weights_by_sender = {}
        self.received_losses_by_sender = {}
        self.noise_values_by_sender = {}
        self.receive_lock = threading.Lock()
        self.local_loss = []

    def getNodeList(self, node_list):
        self.node_list = node_list

    def receiveMessage(self, sender_id, message):
        if len(message.keys()) != 1:
            return
        with self.receive_lock:
            if (
                "ENCRYPTED_WEIGHTS" in message.keys()
                and sender_id in self.node_list.keys()
                and sender_id in self.predecessors
            ):
                self.received_encrypted_weights_by_sender[sender_id] = message[
                    "ENCRYPTED_WEIGHTS"
                ][0]
                self.received_losses_by_sender[sender_id] = list(
                    message["ENCRYPTED_WEIGHTS"][1]
                )

            if "FINAL_NOISE_VALUE" in message.keys() and sender_id in self.node_list:
                self.noise_values_by_sender[sender_id] = message["FINAL_NOISE_VALUE"]
                self.noise_values_count = len(self.noise_values_by_sender)

    def create_ckks_context(self):
        private_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        private_context.global_scale = 2**40
        private_context.generate_galois_keys()

        # The central server retains the secret key. Clients receive only the
        # public/evaluation material required to encrypt and add ciphertexts.
        public_context = ts.context_from(
            private_context.serialize(save_secret_key=False)
        )
        return private_context, public_context

    def decryptWeights(
        self,
        encrypted_weights,
        context,
        original_shapes,
        text_widget,
        client_count,
        noise,
    ):
        start_time = time.time()
        self.network.updateText(
            "Decrypting aggregated weights using CKKS decryption...", text_widget
        )
        decrypted_weights = {}
        weight_size = 0
        for name, enc_weight_chunks in encrypted_weights.items():
            decrypted_flat = []
            for enc_weight in enc_weight_chunks:
                decrypted_flat.extend(enc_weight.decrypt(context.secret_key()))

            decrypted_flat_array = np.array(decrypted_flat, dtype=np.float32)
            weight_size += decrypted_flat_array.nbytes

            decrypted_array = (
                (decrypted_flat_array / client_count) - noise / client_count
            ).reshape(original_shapes[name])
            decrypted_weights[name] = (
                torch.from_numpy(decrypted_array)
                .clone()
                .detach()
                .to(dtype=torch.float32)
            )

            if self.network.checkForNan(
                {name: decrypted_weights[name]}, "Decrypted Weights", text_widget
            ):
                raise ValueError(f"NaN detected in decrypted weights: {name}")

        decryption_time = time.time() - start_time
        self.network.updateText(
            f"Decryption completed in {decryption_time:.4f} seconds.", text_widget
        )
        self.overhead_info["decryption_times"].append(decryption_time)
        return decrypted_weights, weight_size

    def displayNetwork(self, visualisation_canvas=None, visualisation_ax=None):
        nodes = []
        edges = []
        for partition in self.route:
            nodes += partition
            for node in range(1, len(partition)):
                edges.append((partition[node - 1], partition[node]))
        colours = ["red"] * len(nodes)
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        pos = {}
        for i in range(len(self.route)):
            for j in range(len(self.route[i])):
                pos[self.route[i][j]] = (j, -i)
        return colours, pos, graph

    def calculateNoise(self):
        noise_start_time = time.time()
        with self.receive_lock:
            self.noise_values = []
            self.noise_values_by_sender = {}
            self.noise_values_count = 0
        max_noise_count = len(list(self.node_list.keys()))

        self.network.messageAllNodesExcludeServer(0, {"CALC_NOISE": None}, "noise")

        while True:
            with self.receive_lock:
                complete = len(self.noise_values_by_sender) == max_noise_count
            if complete:
                break
            getattr(self.network, "raise_if_failed", lambda: None)()
            time.sleep(0.01)
        with self.receive_lock:
            self.noise_values = [
                self.noise_values_by_sender[client] for client in sorted(self.node_list)
            ]
            self.noise_values_count = len(self.noise_values)
        self.noise_added = math.fsum(self.noise_values)
        self.overhead_info["noise_calc_time"].append(time.time() - noise_start_time)

    def updateOverheadDict(self, epoch_num):
        self.overhead_info["epoch_num"] = epoch_num

        self.overhead_info["noise_calc_num_transmissions"].append(0)
        self.overhead_info["other_num_transmissions"].append(0)

    def trainingProcess(
        self,
        net_glob,
        dataset_train,
        dataset_test,
        dict_party_user,
        stoi,
        tokenizer,
        text_widget,
        visualisation_canvas,
        visualisation_ax,
        overhead_info,
        ax1,
        ax2,
        canvas,
    ):
        self.overhead_info = overhead_info

        net_glob.train()

        epoch_losses = []
        epoch_accuracies = []

        test_dataset = dataset_test

        start_total_time = time.time()

        for iter in range(self.args.epochs):
            self.updateOverheadDict(iter)

            epoch_start_time = time.time()

            key_gen_time = time.time()
            private_context, public_context = self.create_ckks_context()
            self.overhead_info["key_generation_time"].append(time.time() - key_gen_time)

            self.network.updateText(f"+++ Epoch {iter + 1} starts +++", text_widget)

            list_of_nodes = list(self.node_list.keys())
            num_of_partitions = len(list_of_nodes) // self.args.partition_size
            remainder = len(list_of_nodes) % self.args.partition_size
            partition_sizes = np.full(num_of_partitions, self.args.partition_size)

            count = 0
            while remainder != 0:
                if count >= len(partition_sizes):
                    count = 0
                partition_sizes[count] += 1
                remainder -= 1

            random_node_order = list(self.node_list.keys())
            random.shuffle(random_node_order)  # Uses the shared public random stream.
            self.route = []
            self.predecessors = []
            count = 0
            for i in range(num_of_partitions):
                self.route.append(
                    random_node_order[count : count + int(partition_sizes[i])]
                )
                count += int(partition_sizes[i])
            for route in self.route:
                self.predecessors.append(route[-1])

            colours, pos, G = self.displayNetwork(
                visualisation_canvas, visualisation_ax
            )

            original_shapes = {
                name: weight.shape for name, weight in net_glob.state_dict().items()
            }

            self.received_encrypted_weights_list = []
            self.received_encrypted_weights_by_sender = {}
            self.received_losses_by_sender = {}
            self.local_loss = []
            threads = []

            train_time_list = []
            encryption_time_list = []
            aggregate_time_list = []
            weight_size_noise_list = []

            for node_id in list_of_nodes:
                node_object = self.node_list[node_id]
                thread = threading.Thread(
                    target=node_object.client_training,
                    args=(
                        self.route,
                        node_id,
                        dataset_train,
                        dict_party_user,
                        net_glob,
                        stoi,
                        tokenizer,
                        text_widget,
                        public_context,
                        self.overhead_info,
                        train_time_list,
                        encryption_time_list,
                        aggregate_time_list,
                        weight_size_noise_list,
                        G,
                        visualisation_canvas,
                        visualisation_ax,
                        colours,
                        pos,
                    ),
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            getattr(self.network, "raise_if_failed", lambda: None)()
            with self.receive_lock:
                endpoints = [route[-1] for route in self.route]
                missing = set(endpoints) - set(
                    self.received_encrypted_weights_by_sender
                )
                if missing:
                    raise RuntimeError(
                        f"Missing route results from endpoints: {sorted(missing)}"
                    )
                self.received_encrypted_weights_list = [
                    self.received_encrypted_weights_by_sender[client]
                    for client in endpoints
                ]
                self.local_loss = [
                    loss
                    for client in endpoints
                    for loss in self.received_losses_by_sender[client]
                ]

            self.network.updateText(
                "Final clients sending aggregated encrypted weights to server.",
                text_widget,
            )

            received_encrypted_weights = self.received_encrypted_weights_list[0]
            for i in range(1, len(self.received_encrypted_weights_list)):
                aggregateStartTime = time.time()
                received_encrypted_weights = self.network.aggregateEncryptedWeights(
                    received_encrypted_weights,
                    self.received_encrypted_weights_list[i],
                    text_widget,
                )
                aggregate_time_list.append(time.time() - aggregateStartTime)

            self.overhead_info["encryption_times"].append(
                statistics.mean(encryption_time_list)
            )
            self.overhead_info["aggregation_times"].append(
                statistics.mean(aggregate_time_list)
            )
            self.overhead_info["training_times"].append(
                statistics.mean(train_time_list)
            )

            self.overhead_info["weight_size_noise_encryption"].append(
                statistics.mean(weight_size_noise_list)
            )
            self.calculateNoise()

            decrypted_weights, weight_size = self.decryptWeights(
                received_encrypted_weights,
                private_context,
                original_shapes,
                text_widget,
                len(list(self.node_list.keys())),
                self.noise_added,
            )
            self.overhead_info["weight_size_decryption"].append(weight_size)

            if self.network.checkForNan(
                decrypted_weights, "Global Model Weights", text_widget
            ):
                raise ValueError(
                    "NaN detected in global model weights before updating."
                )

            update_start_time = time.time()
            net_glob.load_state_dict(decrypted_weights)
            self.overhead_info["update_times"].append(time.time() - update_start_time)
            self.network.updateText(
                "Server has updated the global model with final aggregated weights.",
                text_widget,
            )

            net_glob.eval()

            test_accuracy, _ = self.evaluator(net_glob, test_dataset, self.args)
            epoch_losses.append(np.mean(self.local_loss))
            epoch_accuracies.append(test_accuracy)
            self.overhead_info["acc_score"].append(test_accuracy)
            self.overhead_info["loss_score"].append(np.mean(self.local_loss))

            self.network.updateText(
                f"Epoch {iter + 1} completed. Test Acc: {test_accuracy:.2f}", text_widget
            )
            self.network.updatePlots(epoch_losses, epoch_accuracies, ax1, ax2, canvas)

            self.overhead_info["epoch_times"].append(time.time() - epoch_start_time)

        total_time_array = np.full(
            shape=self.overhead_info["epoch_num"] + 1,
            fill_value=time.time() - start_total_time,
        )
        self.overhead_info["total_time"] = total_time_array

        return test_accuracy

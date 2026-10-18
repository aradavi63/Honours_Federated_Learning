"""Plaintext FedAvg nodes."""

import copy
import platform
import statistics
import threading
import time

import networkx as nx
import numpy as np
import psutil
import torch

from evaluation.model import test_fun
from training.local import LocalUpdateCNN


def FedAvg(weight_list):
    num = len(weight_list)
    w_avg = copy.deepcopy(weight_list[0])
    for k in w_avg.keys():
        for i in range(1, num):
            w_avg[k] += weight_list[i][k]
        w_avg[k] = torch.div(w_avg[k], num)
    return w_avg


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

    def client_training(
        self,
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
        G,
        visualisation_canvas,
        visualisation_ax,
        colours,
        pos,
    ):
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
        overhead_info["num_transmissions"][overhead_info["epoch_num"]] += 1

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
            f"Client {client_id} local weights before upload",
            text_widget,
        ):
            raise ValueError(
                f"NaN detected in Client {client_id} local weights before upload."
            )

        self.network.messageCentralServer(
            self.node_id, {"RESULTS": [local_weights, loss]}, "other"
        )

        colours[clients_index] = "green"
        self.network.updateDisplayNetwork(
            G, visualisation_canvas, visualisation_ax, colours, pos
        )


class ServerNodeClass:
    def __init__(self, node_id, network, args, evaluator=None):
        super().__init__()
        self.evaluator = evaluator or test_fun
        self.node_id = node_id
        self.network = network
        self.args = args
        self.node_list = []
        self.received_weights_list = []
        self.received_weights_by_client = {}
        self.received_losses_by_client = {}
        self.local_loss = []

    def getNodeList(self, node_list):
        self.node_list = node_list

    def receiveMessage(self, sender_id, message):
        if len(message.keys()) != 1:
            return

        if "RESULTS" in message.keys() and sender_id in self.node_list.keys():
            self.received_weights_by_client[sender_id] = message["RESULTS"][0]
            self.received_losses_by_client[sender_id] = message["RESULTS"][1]

    def displayNetwork(self, visualisation_canvas=None, visualisation_ax=None):
        nodes = list(self.node_list.keys())
        colours = ["red"] * len(nodes)
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        pos = {node: (index, 0) for index, node in enumerate(nodes)}
        return colours, pos, graph

    def updateOverheadDict(self, epoch_num):
        self.overhead_info["epoch_num"] = epoch_num
        self.overhead_info["num_transmissions"].append(0)

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

            context = None

            self.network.updateText(f"+++ Epoch {iter + 1} starts +++", text_widget)

            colours, pos, G = self.displayNetwork(
                visualisation_canvas, visualisation_ax
            )

            self.received_weights_list = []
            self.received_weights_by_client = {}
            self.received_losses_by_client = {}
            self.local_loss = []
            threads = []

            train_time_list = []
            encryption_time_list = []
            aggregate_time_list = []

            for node_id in list(self.node_list.keys()):
                node_object = self.node_list[node_id]
                thread = threading.Thread(
                    target=node_object.client_training,
                    args=(
                        node_id,
                        dataset_train,
                        dict_party_user,
                        net_glob,
                        stoi,
                        tokenizer,
                        text_widget,
                        context,
                        self.overhead_info,
                        train_time_list,
                        encryption_time_list,
                        aggregate_time_list,
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

            ordered_clients = sorted(self.node_list)
            self.received_weights_list = [
                self.received_weights_by_client[client] for client in ordered_clients
            ]
            self.local_loss = [
                self.received_losses_by_client[client] for client in ordered_clients
            ]

            self.network.updateText("Server received client models.", text_widget)

            aggregate_start_time = time.time()
            w_glob = FedAvg(self.received_weights_list)
            aggregate_end_time = time.time() - aggregate_start_time

            self.overhead_info["aggregation_times"].append(aggregate_end_time)
            self.overhead_info["training_times"].append(
                statistics.mean(train_time_list)
            )

            update_start_time = time.time()
            net_glob.load_state_dict(w_glob)
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
                f"Epoch {iter + 1} completed. Test Acc: {test_accuracy:.2f}",
                text_widget,
            )
            self.network.updatePlots(epoch_losses, epoch_accuracies, ax1, ax2, canvas)

            self.overhead_info["epoch_times"].append(time.time() - epoch_start_time)

        total_time_array = np.full(
            shape=self.overhead_info["epoch_num"] + 1,
            fill_value=time.time() - start_total_time,
        )
        self.overhead_info["total_time"] = total_time_array

        return test_accuracy


class NetworkSimulationClass:
    def __init__(self, args):
        self.args = args
        self.nodes = {}
        self.server_node = None
        self.overhead_info = {
            "epoch_num": 0,
            "epoch_times": [],
            "acc_score": [],
            "loss_score": [],
            "total_time": [],
            "training_times": [],
            "num_transmissions": [],
            "key_generation_time": [],
            "encryption_times": [],
            "aggregation_times": [],
            "decryption_times": [],
            "update_times": [],
            "dataset_info": {},
            "system_info": {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "cpu_count": psutil.cpu_count(logical=True),
                "ram_size": psutil.virtual_memory().total // (1024**3),  # in GB
            },
        }

    def messageCentralServer(self, sender_id, message, reason):
        self.checkReason(reason)
        self.server_node.receiveMessage(sender_id, message)

    def checkReason(self, reason):
        if reason == "other":
            self.overhead_info["num_transmissions"][
                self.overhead_info["epoch_num"]
            ] += 1

    def updateText(self, message, text_widget=None):
        return None

    def checkForNan(self, weights_dict, description, text_widget=None):
        for name, weight in weights_dict.items():
            if torch.isnan(weight).any():
                self.updateText(f"NaN detected in {description}: {name}", text_widget)
                return True
        return False

    def updateDisplayNetwork(self, *args, **kwargs):
        return None

    def updatePlots(self, *args, **kwargs):
        return None

    def addNode(self, node):
        if node.node_id == 0:
            self.server_node = node
        else:
            self.nodes[node.node_id] = node

    def getNodes(self):
        return self.nodes

"""SIHEFL message routing and CKKS operations."""

import platform
import threading
import time

import numpy as np
import psutil  
import tenseal as ts
import torch


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
            "noise_calc_time": [],
            "training_times": [],
            "noise_calc_num_transmissions": [],
            "other_num_transmissions": [],
            "key_generation_time": [],
            "encryption_times": [],
            "aggregation_times": [],
            "decryption_times": [],
            "weight_size_noise_encryption": [],
            "weight_size_decryption": [],
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

    def messageSingleNode(self, sender_id, receiver_id, message, reason):
        if sender_id in self.nodes.keys() and receiver_id in self.nodes.keys():
            self.checkReason(reason)
            self.nodes[receiver_id].receiveMessage(sender_id, message)

    def messageCentralServer(self, sender_id, message, reason):
        self.checkReason(reason)
        self.server_node.receiveMessage(sender_id, message)

    def messageAllNodesExcludeServer(self, sender_id, message, reason):
        threads = []
        for key, value in self.nodes.items():
            if key != sender_id:
                self.checkReason(reason)
                t = threading.Thread(
                    target=value.receiveMessage, args=(sender_id, message)
                )
                t.start()
                threads.append(t)

        for t in threads:
            t.join()

    def checkReason(self, reason):
        if reason == "other":
            self.overhead_info["other_num_transmissions"][
                self.overhead_info["epoch_num"]
            ] += 1
        elif reason == "noise":
            self.overhead_info["noise_calc_num_transmissions"][
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

    def encryptWeights(
        self, weights_dict, context, text_widget, noise, chunk_size=1000
    ):
        weight_size_noise = 0
        start_time = time.time()
        self.updateText("Encrypting weights using CKKS encryption...", text_widget)
        encrypted_weights = {}
        for name, weight in weights_dict.items():
            weight_np = weight.cpu().numpy().astype(np.float64)
            if np.isnan(weight_np).any():
                self.updateText(
                    f"NaN detected in {name} before encryption", text_widget
                )
                continue
            weight_size_noise += weight_np.nbytes
            noisey_weight_np = weight_np + noise

            weight_chunks = np.array_split(
                noisey_weight_np.flatten(),
                max(1, len(noisey_weight_np.flatten()) // chunk_size),
            )
            encrypted_chunks = [
                ts.ckks_vector(context, chunk) for chunk in weight_chunks
            ]
            encrypted_weights[name] = encrypted_chunks

        encryption_time = time.time() - start_time
        self.updateText(
            f"Encryption completed in {encryption_time:.4f} seconds.", text_widget
        )
        return encrypted_weights, weight_size_noise

    def aggregateEncryptedWeights(
        self, encrypted_weights1, encrypted_weights2, text_widget
    ):
        start_time = time.time()
        self.updateText(
            "Aggregating encrypted weights using homomorphic addition...", text_widget
        )
        aggregated_weights = {}
        for name in encrypted_weights1.keys():
            aggregated_chunks = []
            for enc_w1, enc_w2 in zip(
                encrypted_weights1[name], encrypted_weights2[name]
            ):
                aggregated_chunk = enc_w1 + enc_w2
                aggregated_chunks.append(aggregated_chunk)
            aggregated_weights[name] = aggregated_chunks

        aggregation_time = time.time() - start_time
        self.updateText(
            f"Encrypted weights aggregation completed in {aggregation_time:.4f} seconds.",
            text_widget,
        )
        return aggregated_weights

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

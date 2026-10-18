"""Client-side data poisoning shared by different schemes."""

import threading
from collections import Counter

from attacks.poisoning.label_flip import (
    LabelFlipDataset,
    choose_poisoned_indices,
    count_source_records,
)


class PoisoningTransform:
    def __init__(self, attack, partitions):
        self.attack = attack
        self.partitions = partitions
        self.round = 0
        self.selected = {}
        self.source_counts = {}
        self.active = set()
        self.label_reads = Counter()
        self._lock = threading.Lock()

    def prepare(self, dataset):
        """Count source-class records and select records to poison."""
        a = self.attack
        self.source_counts = {
            c: count_source_records(dataset, ids, a.mapping.source_class)
            for c, ids in self.partitions.items()
        }
        for client in a.malicious_client_ids:
            if client not in self.partitions:
                raise ValueError("Unknown malicious client")
            self.selected[client] = choose_poisoned_indices(
                dataset,
                self.partitions[client],
                a.mapping.source_class,
                a.poison_fraction,
                a.seed + client,
            )

    def begin_round(self, round_number):
        self.round = round_number
        self.active = set()
        self.label_reads = Counter()

    def __call__(self, client, dataset):
        a = self.attack
        if not a.active_at(self.round) or client not in a.malicious_client_ids:
            return dataset
        if client not in self.selected:
            self.selected[client] = choose_poisoned_indices(
                dataset,
                self.partitions[client],
                a.mapping.source_class,
                a.poison_fraction,
                a.seed + client,
            )
        if not self.selected[client]:
            return dataset
        self.active.add(client)

        def record_flip(index, source, target):
            with self._lock:
                self.label_reads[(client, int(index), source, target)] += 1

        return LabelFlipDataset(
            dataset, a.mapping, self.selected[client], on_flip=record_flip
        )

    def audit(self):
        """Actual labels returned by the client dataset; not DP optimiser-use counts."""
        return [
            dict(
                round=self.round,
                client=c,
                dataset_index=i,
                source=s,
                target=t,
                label_reads=n,
            )
            for (c, i, s, t), n in sorted(self.label_reads.items())
        ]

    def summary(self):
        return {
            "eligible_selected_records": sum(map(len, self.selected.values())),
            "zero_exposure": not any(self.selected.values()),
            "active_poisoning_clients": len(self.active),
            "observed_flipped_records": len(self.label_reads),
            "flipped_label_reads": sum(self.label_reads.values()),
            "flipped_source_records": sum(len(self.selected[c]) for c in self.active),
        }

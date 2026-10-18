import random
from collections import defaultdict

import numpy as np
import torch


def build_classes_dict(dataset, args):
    class_dict = defaultdict(list)
    for ind, x in enumerate(dataset):
        _, label = x
        if torch.is_tensor(label):
            label = int(label.item())
        if label in class_dict:
            class_dict[label].append(ind)
        else:
            class_dict[label] = [ind]
    return class_dict


def sample_dirichlet_train_data(dataset, num_participants, num_samples, args):
    if getattr(args, "iid", False):
        chunks = np.array_split(np.random.permutation(len(dataset)), num_participants)
        full = {client: chunk.tolist() for client, chunk in enumerate(chunks)}
        count = min(num_samples, min(map(len, chunks)))
        sampled = {
            client: chunk[:count].tolist() for client, chunk in enumerate(chunks)
        }
        return full, sampled
    data_classes = build_classes_dict(dataset, args)
    class_size = len(data_classes[0])
    per_participant_list = defaultdict(list)
    per_samples_list = defaultdict(list)
    no_classes = len(data_classes.keys())
    for n in range(no_classes):
        random.shuffle(data_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(num_participants * [args.alpha])
        )
        for user in range(num_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = data_classes[n][: min(len(data_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            data_classes[n] = data_classes[n][min(len(data_classes[n]), no_imgs) :]

    for i in range(len(per_participant_list)):
        num_samples = min(num_samples, len(per_participant_list[i]))

    for i in range(len(per_participant_list)):
        sample_index = np.random.choice(
            len(per_participant_list[i]), num_samples, replace=False
        )
        per_samples_list[i].extend(np.array(per_participant_list[i])[sample_index])

    return per_participant_list, per_samples_list

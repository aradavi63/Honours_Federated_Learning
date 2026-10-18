"""Public test-set confusion matrix and targeted poisoning metrics."""

import torch
from torch.utils.data import DataLoader


def calculate_confusion_matrix(net, dataset, args):
    confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False)
    net.to(args.device)
    net.eval()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(args.device)
            predictions = net(inputs).argmax(dim=1).cpu()
            labels = labels.to(dtype=torch.int64).cpu()
            bins = torch.bincount(
                labels * args.num_classes + predictions,
                minlength=args.num_classes * args.num_classes,
            )
            confusion += bins.reshape(args.num_classes, args.num_classes)
    return confusion


def mapping_metrics(confusion, mapping):
    source_total = int(confusion[mapping.source_class].sum().item())
    source_correct = int(confusion[mapping.source_class, mapping.source_class].item())
    source_to_target = int(confusion[mapping.source_class, mapping.target_class].item())
    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())

    target = mapping.target_class
    target_total = int(confusion[target].sum())
    target_correct = int(confusion[target, target])
    remaining = [
        i for i in range(len(confusion)) if i not in (mapping.source_class, target)
    ]
    remaining_recalls = [
        _percentage(int(confusion[i, i]), int(confusion[i].sum())) for i in remaining
    ]
    return {
        "target_recall": _percentage(target_correct, target_total),
        "remaining_class_macro_recall": sum(remaining_recalls) / len(remaining_recalls),
        "remaining_class_accuracy": _percentage(
            correct - source_correct - target_correct,
            total - source_total - target_total,
        ),
        "source_class": mapping.source_class,
        "source_name": mapping.source_name,
        "target_class": mapping.target_class,
        "target_name": mapping.target_name,
        "source_recall": _percentage(source_correct, source_total),
        "source_to_target_asr": _percentage(source_to_target, source_total),
        "source_total": source_total,
        "source_correct": source_correct,
        "source_to_target_count": source_to_target,
        "other_class_accuracy": _percentage(
            correct - source_correct, total - source_total
        ),
    }


def _percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return 100.0 * numerator / denominator

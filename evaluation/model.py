"""Classification accuracy and cross-entropy evaluation."""

import torch
import torch.nn.functional
from torch.utils.data import DataLoader


def test_fun(net, dataset, args):
    """Return accuracy in percent and mean cross-entropy per example."""
    net.to(args.device)
    net.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(
        dataset, batch_size=getattr(args, "bs", args.local_bs), shuffle=False
    )

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = net(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels, reduction="sum")
            test_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.0 * correct / len(data_loader.dataset)
    return accuracy, test_loss

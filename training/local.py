import torch
from torch.utils.data import DataLoader, Dataset


class LocalUpdateCNN:
    def __init__(self, args, dataset, idxs, generator=None):
        self.args = args
        split_dataset = DatasetSplitCNN(dataset, idxs)
        self.ldr_train = DataLoader(
            split_dataset,
            batch_size=self.args.local_bs,
            shuffle=True,
            generator=generator,
        )

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=getattr(self.args, "momentum", 0.5),
        )
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = (
                    images.to(self.args.device),
                    labels.to(self.args.device),
                )
                net.zero_grad()
                outputs = net(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class DatasetSplitCNN(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

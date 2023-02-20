import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def vectorise(batch):
    batch_size = batch.size(0)
    return batch.reshape(batch_size, -1).squeeze()


def one_hot(labels, n_classes=10):
    arr = torch.eye(n_classes)
    return arr[labels]


class MNIST(datasets.MNIST):
    def __init__(self, train, path="./data", normalise=True):
        if normalise:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(path, download=True, transform=transform, train=train)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = vectorise(img)
        label = one_hot(label)
        return img, label

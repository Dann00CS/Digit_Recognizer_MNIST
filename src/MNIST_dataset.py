import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class MNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        self.dataset = datasets.MNIST(root=self.root, train=self.train, download=True)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.dataset)
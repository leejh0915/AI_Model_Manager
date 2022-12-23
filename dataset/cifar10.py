import os, sys

import torch
import torchvision
import torchvision.transforms as transforms


class cifar10:
    def __init__(self):
        # print('test')
        # print(os.path.dirname(os.path.abspath(__file__)))

        # Image preprocessing modules
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()])

        # CIFAR-10 dataset
        self.train_dataset = torchvision.datasets.CIFAR10(root='dataset/data',
                                                          train=True,
                                                          transform=transform,
                                                          download=True)

        self.test_dataset = torchvision.datasets.CIFAR10(root='dataset/data',
                                                         train=False,
                                                         transform=transforms.ToTensor())

        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=100,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=100,
                                                       shuffle=False)

    def set_test_dataset(self):
        return self.test_dataset, self.test_loader
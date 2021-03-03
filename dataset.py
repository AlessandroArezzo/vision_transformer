import glob
import os
import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler
from PIL import Image
import torch
import torchvision

class CustomDataset(Dataset):
    def __init__(self, images, labels, mode='train'):
        assert mode == 'train' or mode == 'test'
        self.files_images = glob.glob(os.path.join(images, mode) + '/*.*')
        self.labels = labels

    def __getitem__(self, index):
        return Image.open(self.files_images[index % len(self.files_images)]), self.labels[index % len(self.files_images)]

    def __len__(self):
        return len(self.files_images)

class LoadTorchData():
    def __init__(self, root_path=None, download=True):
        self.root_path = root_path
        self.download = download

    def load_dataset(self, dataset_name, batch_size_train=100, batch_size_test=1000, val_ratio=0.3, n_cpu=8):
        assert dataset_name == "CIFAR-10" or dataset_name == "CIFAR-100"
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        self.root_path = os.path.join(self.root_path, dataset_name)
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        if dataset_name == "CIFAR-10":
            train_dataset = torchvision.datasets.CIFAR10(self.root_path, train=True, download=self.download,
                                                   transform=transform)
            test_set = torchvision.datasets.CIFAR10(self.root_path, train=False, download=self.download,
                                                    transform=transform)
        elif dataset_name == "CIFAR-100":
            train_dataset = torchvision.datasets.CIFAR100(self.root_path, train=True, download=self.download,
                                                   transform=transform)
            test_set = torchvision.datasets.CIFAR100(self.root_path, train=False, download=self.download,
                                                     transform=transform)
        dataset_len = len(train_dataset)
        indices = list(range(dataset_len))
        val_len = int(np.floor(val_ratio * dataset_len))
        validation_idx = np.random.choice(indices, size=val_len, replace=False)
        train_idx = list(set(indices) - set(validation_idx))

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, sampler=train_sampler,
                                                   num_workers=n_cpu)
        validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, sampler=validation_sampler,
                                                    num_workers=n_cpu)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True,
                                                    num_workers=n_cpu)
        return train_loader, validation_loader, test_loader

        """
        if dataset_name == "CIFAR-10":
            train_set = torchvision.datasets.CIFAR10(self.root_path, train=True, download=self.download,
                                                     transform=transform)
            test_set = torchvision.datasets.CIFAR10(self.root_path, train=False, download=self.download,
                                                    transform=transform)
        elif dataset_name == "CIFAR-100":
            train_set = torchvision.datasets.CIFAR100(self.root_path, train=True, download=self.download,
                                                      transform=transform)
            test_set = torchvision.datasets.CIFAR100(self.root_path, train=False, download=self.download,
                                                     transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True)
        return train_loader, test_loader
        """

import os
from torch.utils.data import Dataset
import utils
from PIL import Image
import scipy.io
import torch
import torchvision

class CustomDataset(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.files_images, self.labels = self.get_images_and_labels()

    def __getitem__(self, index):
        return self.transforms(Image.open(self.files_images[index % len(self.files_images)]).convert('RGB')), self.labels[index % len(self.files_images)]

    def __len__(self):
        return len(self.files_images)

    def get_images_and_labels(self):
        raise NotImplementedError("To implement in child class")

class OxfordPetsDataset(CustomDataset):
    def __init__(self, dataset_path, mode='train', transforms=None):
        assert mode == 'train' or mode == 'test', "data loader mode must be train or test"
        self.dataset_path = dataset_path
        self.mode = mode
        super().__init__(transforms)

    def get_images_and_labels(self):
        images_path = os.path.join(self.dataset_path, "images")
        labels_path = os.path.join(self.dataset_path, "labels")
        split_file_name = "trainval.txt"
        if self.mode == "test":
            split_file_name = "test.txt"
        split_file_path = os.path.join(labels_path, split_file_name)
        split_file = open(split_file_path, 'r')
        line = split_file.readlines()
        image_files, labels = [], []
        for line in line:
            image_name, label = line.split(" ")[0]+".jpg", line.split(" ")[1]
            image_files.append(os.path.join(images_path, image_name))
            labels.append(int(label)-1)
        return image_files, labels

class OxfordFlowersDataset(CustomDataset):
    def __init__(self, dataset_path, mode='train', transforms=None):
        assert mode == 'train' or mode == "val" or mode == 'test', "data loader mode must be train, val or test"
        assert os.path.isdir(dataset_path), "Dataset not found in path '"+dataset_path+"'"
        self.dataset_path = dataset_path
        self.mode = mode
        super().__init__(transforms)

    def get_images_and_labels(self):
        images_path = os.path.join(self.dataset_path, "Oxford-III-Flowers-102", "102flowers", "jpg")
        labels_path = os.path.join(self.dataset_path, "Oxford-III-Flowers-102")
        if self.mode == "train":
            key_setid = "tstid"
        elif self.mode == "val":
            key_setid = "valid"
        else:
            key_setid = "trnid"
        labels_file_path = os.path.join(labels_path, "imagelabels.mat")
        split_file_path = os.path.join(labels_path, "setid.mat")
        labels_file = scipy.io.loadmat(labels_file_path)
        split_file = scipy.io.loadmat(split_file_path)

        all_labels, set_ids = labels_file["labels"][0], split_file[key_setid][0]
        image_files, labels = [], []
        for id in set_ids:
            image_name, label = 'image_%05d.jpg' % id, all_labels[id-1]
            image_files.append(os.path.join(images_path, image_name))
            labels.append(int(label)-1)
        return image_files, labels

class LoadTorchData():
    def __init__(self, root_path=None, download=True):
        self.root_path = root_path
        self.download = download

    def load_dataset(self, dataset_name, batch_size_train=100, batch_size_test=1000, val_ratio=0.3, n_cpu=8, transforms=None):
        assert dataset_name == "CIFAR-10" or dataset_name == "CIFAR-100", \
            "Pytorch datasets permitted are: CIFAR-10, CIFAR-100"
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        split_train_val = False
        train_dataset, val_dataset, test_dataset = None, None, None
        if dataset_name == "CIFAR-10":
            train_dataset = torchvision.datasets.CIFAR10(self.root_path, train=True, download=self.download,
                                                   transform=transforms["train"])
            test_dataset = torchvision.datasets.CIFAR10(self.root_path, train=False, download=self.download,
                                                    transform=transforms["test"])
            split_train_val = True
        elif dataset_name == "CIFAR-100":
            train_dataset = torchvision.datasets.CIFAR100(self.root_path, train=True, download=self.download,
                                                   transform=transforms["train"])
            test_dataset = torchvision.datasets.CIFAR100(self.root_path, train=False, download=self.download,
                                                     transform=transforms["test"])
            split_train_val = True
        if split_train_val:
            train_loader, validation_loader, test_loader = utils.get_loader_splitting_val(train_dataset=train_dataset,
                                                    test_dataset=test_dataset, batch_size_train=batch_size_train,
                                                    batch_size_test=batch_size_test, n_cpu=n_cpu, val_ratio=val_ratio)
        else:
            train_loader, validation_loader, test_loader = self.__get_loader_from_separate_dataset(
                                                                                           train_dataset=train_dataset,
                                                                                           val_dataset=val_dataset,
                                                                                           test_dataset=test_dataset,
                                                                                           batch_size_train=batch_size_train,
                                                                                           batch_size_test=batch_size_test,
                                                                                           n_cpu=n_cpu)

        return train_loader, validation_loader, test_loader


    def __get_loader_from_separate_dataset(self, train_dataset, val_dataset, test_dataset, batch_size_train,
                                           batch_size_test, n_cpu,):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train,
                                                   sampler=train_dataset,
                                                   num_workers=n_cpu)
        validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train,
                                                        sampler=val_dataset,
                                                        num_workers=n_cpu)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True,
                                                  num_workers=n_cpu)
        return train_loader, validation_loader, test_loader

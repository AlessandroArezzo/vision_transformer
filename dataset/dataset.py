import os

from torch.utils.data import Dataset
import utils
from PIL import Image
import torchvision

class CustomDataset(Dataset):
    """
    This class contains an abstract implementation of a dataset not included in the torchvision library.
    Is required the definition of the method get_images_and_labels in each concrete classes.

    Attributes:
    transforms        reference to an object of class torchvision.transforms.Compose that apply transformation
                      to the images
    files_images      list that must contains all image file path
    labels            list that must contains the labels of the image files
    """

    def __init__(self, transforms=None):
        self.transforms = transforms
        self.files_images, self.labels = self.get_images_and_labels()

    def __getitem__(self, index):
        return self.transforms(Image.open(self.files_images[index]).convert('RGB')), self.labels[index]

    def __len__(self):
        return len(self.files_images)

    def get_images_and_labels(self):
        raise NotImplementedError("To implement in child class")

class OxfordPetsDataset(CustomDataset):
    """
    This class implements the Oxford-IIIT Pets dataset

    Attributes:
    dataset_path      string that contains the dataset path
    mode              string that determines whether to extract train or test data
    """

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
            image_name, label = line.split(" ")[0]+".jpg", line.split(" ")[2]
            image_files.append(os.path.join(images_path, image_name))
            labels.append(int(label)-1)
        return image_files, labels

class LoadTorchData():
    """
    This class download a dataset from torchvision library and return the relative torch data loaders.

    Attributes:
    root_path         string that contains the dataset path
    download          boolean that determines if the dataset must be saved in the root_path or it is already present
    """

    def __init__(self, root_path=None, download=True):
        self.root_path = root_path
        self.download = download

    def load_dataset(self, dataset_name, batch_size_train=100, batch_size_test=1000, val_ratio=0.3, n_cpu=8, transforms=None):
        """
        Method that return the data loaders of a torchvision dataset
        :param dataset_name: name of the torchvision dataset
               batch_size_train: batch size to use for train data loader
               batch_size_test:  batch size to use for test data loader
               val_ratio: percent of the data train to use for validation data loader
               n_cpu: num workers to use
               transforms: transformation to apply
        :return: train, validation and test torch data loader
        """
        assert dataset_name == "CIFAR-10" or dataset_name == "CIFAR-100", \
            "Pytorch datasets permitted are: CIFAR-10, CIFAR-100"
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)
        train_dataset, test_dataset = None, None
        if dataset_name == "CIFAR-10":
            train_dataset = torchvision.datasets.CIFAR10(self.root_path, train=True, download=self.download,
                                                   transform=transforms["train"])
            test_dataset = torchvision.datasets.CIFAR10(self.root_path, train=False, download=self.download,
                                                    transform=transforms["test"])
        elif dataset_name == "CIFAR-100":
            train_dataset = torchvision.datasets.CIFAR100(self.root_path, train=True, download=self.download,
                                                   transform=transforms["train"])
            test_dataset = torchvision.datasets.CIFAR100(self.root_path, train=False, download=self.download,
                                                     transform=transforms["test"])
        train_loader, validation_loader, test_loader = utils.get_loader(train_dataset=train_dataset,
                                                test_dataset=test_dataset, batch_size_train=batch_size_train,
                                                batch_size_test=batch_size_test, n_cpu=n_cpu, val_ratio=val_ratio)
        return train_loader, validation_loader, test_loader

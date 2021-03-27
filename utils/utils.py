import csv
import os

from matplotlib import pyplot as plt
from torch.utils.data import SubsetRandomSampler, DataLoader
import numpy as np
from PIL import Image
import torchvision
import pandas as pd

from models.ViT_model import ViT

from models.hybrid_ViT_model import Resnet18HybridViT
from dataset.dataset import OxfordPetsDataset
from dataset.dataset import LoadTorchData

def get_ViT_name(model_type, patch_size=16, hybrid=False):
    """
    Method that return the name of a ViT configuration
    :param model_type: ViT-XS or ViT-S
           patch_size: dimensions of the patch size in which to partition the images
           hybrid: define if the model is an hybrid configuration or not
    :return: name of ViT
    """
    if hybrid:
        model_name = "resnet18+"+str(model_type)
    else:
        model_name = str(model_type)+"_"+str(patch_size)
    return model_name

def get_ViT_model(type, image_size, patch_size, n_classes, n_channels, dropout, hybrid=False):
    """
    Method that return the ViT model given its parameters
    :param type: ViT-XS or ViT-S
           image_size: images resolution
           patch_size: dimensions of the patch size in which to partition the images
           n_classes: number of classes
           n_channels: number of image channels
           dropout: dropout to use
           hybrid: define if to use hybrid model or not
    :return: ViT model
    """
    assert type == "ViT-XS" or type == "ViT-S" or type == "ViT-XXS", \
        "ViT type error: type permitted are 'ViT-XS', 'ViT-S'"
    if type == "ViT-XS":
        emb_dim, n_heads, depth, mlp_size = 128, 8, 8, 384
    elif type == "ViT-S":
        emb_dim, n_heads, depth, mlp_size = 256, 8, 10, 768
    if hybrid:
        model = Resnet18HybridViT(image_size=image_size, num_classes=n_classes,
                    dim=emb_dim, depth=depth, num_heads=n_heads,
                    feedforward_dim=mlp_size, dropout=dropout)
    else:
        model = ViT(image_size=image_size, patch_size=patch_size, num_classes=n_classes,
                   channels=n_channels, dim=emb_dim, depth=depth, num_heads=n_heads,
                   feedforward_dim=mlp_size, dropout=dropout)
    return model

def get_resnet_model(resnet_type, n_classes):
    """
    Method that return the resnet model
    :param type: resnet18 or resnet34
           n_classes: number of classes
    :return: resnet model
    """
    assert resnet_type == "resnet18" or resnet_type == "resnet34", \
        "resnet type error: type permitted are 'resnet18', 'resent34'"
    if resnet_type == "resnet18":
        model = torchvision.models.resnet18(pretrained=False, num_classes=n_classes)
    elif resnet_type == "resnet34":
        model = torchvision.models.resnet34(pretrained=False, num_classes=n_classes)
    return model

def get_output_path(root_path, model_name, dataset_name):
    """
    Method that return the path for the graph and model output
    :param root_path: path of the root directory
           model_name: name of the model to train
           dataset_name: name of the dataset to use
    :return: paths of the directories where to save the graphs generated and the model trained
    """
    root_path = os.path.join(root_path, dataset_name, model_name)
    graph_path = os.path.join(root_path, "graph")
    model_path = os.path.join(root_path, "model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    model_path = os.path.join(model_path, model_name+".pth")
    return graph_path, model_path

def get_transforms(augmentation, image_size):
    """
    Method that return the transformation to apply to the images.
    :param augmentation: define if use data augmentation techniques for the train and validation loader
           image_size: images resolution
    :return: dict of references to objects of the torchvision.transforms.Compose class. Dict contains transform to apply
            for training, validation and test data loader
    """
    if augmentation:
        transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.Resize((int(image_size*1.12), int(image_size*1.12)), Image.BICUBIC),
                torchvision.transforms.RandomCrop(image_size, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize((int(image_size*1.12), int(image_size*1.12)), Image.BICUBIC),
                torchvision.transforms.RandomCrop(image_size, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        }
    else:
        transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.Resize((image_size, image_size), Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize((image_size, image_size), Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        }
    transforms['test'] = torchvision.transforms.Compose([
                torchvision.transforms.Resize((image_size, image_size), Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    return transforms

def get_loader_from_dataset(dataset_name, root_path, image_size, batch_size_train=128, batch_size_test=128,
                              augmentation=True, val_ratio=0.2, n_cpu=8):
    """
    Method that return the torch data loader
    :param dataset_name: name of the dataset to use
           root_path: path of the root directory
           image_size: images resolution
           batch_size_train: batch size to use for train data loader
           batch_size_test:  batch size to use for test data loader
           augmentation: define if use data augmentation
           val_ratio: percent of the data train to use for validation data loader
           n_cpu: num workers to use
    :return: torch loader for train, validation and test data
    """
    assert dataset_name == "oxfordpets" or dataset_name == "CIFAR-10" \
           or dataset_name == "CIFAR-100",\
        "custom dataset name error: you can actually select oxfordpets, oxfordflowers, CIFAR-10 or CIFAR-100"
    train_loader, val_loader, test_loader = None, None, None
    transforms = get_transforms(augmentation=augmentation, image_size=image_size)
    dataset_path = os.path.join(root_path, dataset_name)
    if dataset_name == "oxfordpets":
        trainval_dataset = OxfordPetsDataset(dataset_path=dataset_path, mode="train", transforms=transforms["train"])
        test_dataset = OxfordPetsDataset(dataset_path=dataset_path, mode="test", transforms=transforms["test"])
        train_loader, val_loader, test_loader = get_loader(train_dataset=trainval_dataset,
                                        test_dataset=test_dataset, batch_size_train=batch_size_train,
                                        batch_size_test=batch_size_test, n_cpu=n_cpu, val_ratio=val_ratio)
    elif dataset_name == "CIFAR-10" or dataset_name == "CIFAR-100":
        train_dataset, test_dataset = LoadTorchData(root_path=dataset_path, download=True).load_dataset(
                                                        dataset_name, transforms)
        train_loader, validation_loader, test_loader = get_loader(train_dataset=train_dataset,
                                                test_dataset=test_dataset, batch_size_train=batch_size_train,
                                                batch_size_test=batch_size_test, n_cpu=n_cpu, val_ratio=val_ratio)
    return train_loader, val_loader, test_loader

def get_loader(train_dataset, test_dataset, batch_size_train, batch_size_test, n_cpu, val_ratio):
    """
    Method that return torch data loader from train and test custom dataset object.
    :param train_dataset: train custom dataset object
           test_dataset: test cistom dataset object
           batch_size_train: batch size to use for train data loader
           batch_size_test:  batch size to use for test data loader
           n_cpu: num workers to use
           val_ratio: percent of the data train to use for validation data loader
    :return: torch loader for train, validation and test data
    """
    dataset_len = len(train_dataset)
    indices = list(range(dataset_len))
    val_len = int(np.floor(val_ratio * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, sampler=train_sampler,
                                               num_workers=n_cpu)
    validation_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                                                    sampler=validation_sampler,
                                                    num_workers=n_cpu)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, num_workers=n_cpu, shuffle=True)
    return train_loader, validation_loader, test_loader

def read_csv_from_path(csv_path):
    """
    Method that read a csv from path.
    :param csv_path: path of the csv file
    :return: dict that contains the data read
    """
    data = {}
    with open(csv_path, 'r') as data_file:
        reader = csv.reader(data_file)
        for idx, row in enumerate(reader):
            if idx > 0:
                dataset = row[0]
                model = row[1]

                if dataset not in data.keys():
                    data[dataset] = {}
                data[dataset][model] = {'#epochs': row[2], 'batch_size': row[3], 'lr': row[4], 'optimizer': row[5],
                                        'dropout': row[6], 'test_loss': row[7], 'test_acc': row[8], 'epoch': row[9],
                                        'best_time': row[10], 'exec_time': row[11]}
    return data

def write_on_csv(data, out_df, csv_path):
    """
    Method that write data into a csv
    :param data: data to write in csv
           out_df: data for the csv header
           csv_path: path where to write the csv file
    :return:
    """
    for dataset in data.keys():
        for model in data[dataset].keys():
            data_to_add = [dataset, model, data[dataset][model]["#epochs"],  data[dataset][model]["batch_size"],
                           data[dataset][model]["lr"], data[dataset][model]["optimizer"],
                           data[dataset][model]["dropout"], data[dataset][model]["test_loss"],
                           data[dataset][model]["test_acc"], data[dataset][model]["epoch"],
                           data[dataset][model]["best_time"], data[dataset][model]["exec_time"]]
            data_df_scores = np.hstack((np.array(data_to_add).reshape(1, -1)))
            out_df = out_df.append(pd.Series(data_df_scores.reshape(-1), index=out_df.columns),
                                   ignore_index=True)
    out_df.to_csv(csv_path, index=False, header=True)

def get_time_in_format(millisecond):
    """
    Method that returns time in a readable format string.
    :param millisecond: time to print
    :return: string that contains time in a readable format
    """
    hours, rem = divmod(millisecond, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

def save_result_on_csv(csv_path, model_name, dataset_name, batch_size,
                       lr, n_epochs, execution_time, optimizer, dropout, overwrite=False, best_test_loss=0, best_test_acc=0, best_epoch=0,
                       best_time=0):
    """
    Method that save currently training result in a csv file.
    :param csv_path: path pof the csv where to write results
           ...other data to save
    :return:
    """
    data = {}
    if not overwrite and os.path.isfile(csv_path):
        data = read_csv_from_path(csv_path)
    out_df_scores = pd.DataFrame(columns=['dataset', 'model', '#epochs', 'batch_size', 'lr', 'optimizer', 'dropout',
                                          'test_loss', 'test_acc(%)', 'epoch', 'best_time(h)', 'training_time(h)'])
    if dataset_name not in data.keys():
        data[dataset_name] = {}
    execution_time = get_time_in_format(execution_time)
    best_time = get_time_in_format(best_time)
    data[dataset_name][model_name] = {'#epochs': str(n_epochs), 'batch_size': str(batch_size), 'lr': str(lr),
                                      'optimizer': optimizer, 'dropout': str(dropout),
                                      'test_loss': str( '{:.4f}'.format(best_test_loss)),
                                      'test_acc': '{:4.2f}'.format(best_test_acc), 'epoch': str(best_epoch),
                                       'best_time': best_time, 'exec_time': execution_time}
    write_on_csv(data, out_df_scores, csv_path)

def update_graph(train_loss_history, val_loss_history, train_acc_history, val_acc_history, path):
    """
    Method that update graphs with the trend of loss and accuracy on the train and validation data
    :param train_loss_history, val_loss_history: list that contains loss value for train and validation detected
                                                 at each epoch
           train_acc_history, val_loss_hystory: list that contains mean accuracy for train and validation
                                                detected at each epoch
           path: path of the directory where to save the graphs
    :return:
    """
    losses_img_file = os.path.join(path, "pre_training_losses.png")
    acc_img_file = os.path.join(path, "pre_training_accuracy.png")
    epochs = np.arange(1, len(train_loss_history)+1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Plot Training/Validation Losses")
    plt.ylim(0, max(max(train_loss_history), max(val_loss_history)))
    plt.plot(epochs, train_loss_history, label="average train loss")
    plt.plot(epochs, val_loss_history, label="average validation loss")
    plt.legend()
    plt.savefig(losses_img_file)
    plt.close()
    plt.title("Plot Validation Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.ylim(0, 100)
    plt.plot(epochs, train_acc_history, label="average train accuracy")
    plt.plot(epochs, val_acc_history, label="average validation accuracy")
    plt.legend()
    plt.savefig(acc_img_file)
    plt.close()
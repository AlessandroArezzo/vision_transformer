import os
from torch.nn.functional import nll_loss
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch import no_grad, log_softmax, max
import numpy as np
from PIL import Image
from dataset import LoadTorchData
import torchvision

from models.ViT_model import ViT
from models.hybrid_ViT_model import ResnetHybridViT
from dataset import OxfordPetsDataset, OxfordFlowersDataset

def get_ViT_model(type, image_size, patch_size, n_classes, n_channels, dropout, hybrid, backbone_name):
    assert type == "XSmall" or type == "Small" or type == "Base" or type == "Large" or type == "Huge", \
        "ViT type error: type permitted are 'XSmall', 'Small', 'Base', Large' and 'Huge'"
    if type == "XSmall":
        emb_dim, n_heads, depth, mlp_size = 32, 4, 4, 64
    elif type == "Small":
        emb_dim, n_heads, depth, mlp_size = 64, 8, 8, 128
    elif type == "Base":
        emb_dim, n_heads, depth, mlp_size = 768, 12, 12, 3072
    elif type == "Large":
        emb_dim, n_heads, depth, mlp_size = 1024, 16, 24, 4096
    elif type == "Huge":
        emb_dim, n_heads, depth, mlp_size = 1280, 16, 32, 5120
    if hybrid:
        model = ResnetHybridViT(image_size=image_size, num_classes=n_classes,
                    dim=emb_dim, depth=depth, num_heads=n_heads,
                    feedforward_dim=mlp_size, dropout=dropout, backbone=backbone_name)
    else:
        model = ViT(image_size=image_size, patch_size=patch_size, num_classes=n_classes,
                   channels=n_channels, dim=emb_dim, depth=depth, num_heads=n_heads,
                   feedforward_dim=mlp_size, dropout=dropout)
    return model

def get_output_path(root_path, data_augmentation, dropout, model_type, patch_size, dataset_name, hybrid, backbone_name):
    root_path = os.path.join(root_path, dataset_name)
    if hybrid:
        root_path = os.path.join(root_path, backbone_name+"ViT-"+str(model_type)+"_"+str(patch_size))
    else:
        root_path = os.path.join(root_path, "ViT-"+str(model_type)+"_"+str(patch_size))
    if data_augmentation and dropout > 0:
        root_path = os.path.join(root_path, "augmentation_dropout")
    elif data_augmentation:
        root_path = os.path.join(root_path, "augmentation")
    elif dropout > 0:
        root_path = os.path.join(root_path, "dropout")
    else:
        root_path = os.path.join(root_path, "natural")
    graph_path = os.path.join(root_path, "graph")
    model_path = os.path.join(root_path, "pretrained_models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    model_path = os.path.join(model_path, "ViT-"+str(model_type)+"_"+str(patch_size)+".pth")
    return graph_path, model_path

def get_transforms(augmentation, image_size):
    if augmentation:
        transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(int(image_size)), Image.BICUBIC),
                torchvision.transforms.RandomCrop(image_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(int(image_size)), Image.BICUBIC),
                torchvision.transforms.RandomCrop(image_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        }
    else:
        transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(int(image_size)), Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(int(image_size)), Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        }
    transforms['test'] = torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(int(image_size)), Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transforms

def get_loader_from_dataset(dataset_name, root_path, image_size, batch_size_train=1, batch_size_test=1,
                              augmentation=True, val_ratio=0.2, n_cpu=8):
    assert dataset_name == "oxfordpets" or dataset_name == "oxfordflowers" or dataset_name == "CIFAR-10" \
           or dataset_name == "CIFAR-100",\
        "custom dataset name error: you can actually select oxfordpets, oxfordflowers, CIFAR-10 or CIFAR-100"
    train_loader, val_loader, test_loader = None, None, None
    transforms = get_transforms(augmentation=augmentation, image_size=image_size)
    dataset_path = os.path.join(root_path, dataset_name)
    if dataset_name == "oxfordpets":
        trainval_dataset = OxfordPetsDataset(dataset_path=dataset_path, mode="train",
                                             transforms=transforms["train"])
        test_dataset = OxfordPetsDataset(dataset_path=dataset_path, mode="test",
                                            transforms=transforms["test"])
        train_loader, val_loader, test_loader = get_loader_splitting_val(train_dataset=trainval_dataset,
                                        test_dataset=test_dataset, batch_size_train=batch_size_train,
                                        batch_size_test=batch_size_test, n_cpu=n_cpu, val_ratio=val_ratio)
    elif dataset_name == "oxfordflowers":
        train_dataset = OxfordFlowersDataset(dataset_path=dataset_path, mode="train",
                                             transforms=transforms["train"])
        val_dataset = OxfordFlowersDataset(dataset_path=dataset_path, mode="val",
                                             transforms=transforms["val"])
        test_dataset = OxfordFlowersDataset(dataset_path=dataset_path, mode="test",
                                            transforms=transforms["test"])
        train_loader = DataLoader(train_dataset, batch_size=batch_size_train,num_workers=n_cpu)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_train, num_workers=n_cpu)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, num_workers=n_cpu)
    elif dataset_name == "CIFAR-10" or dataset_name == "CIFAR-100":
        return LoadTorchData(root_path=dataset_path, download=True).load_dataset(dataset_name, batch_size_train,
                                                                                  batch_size_test, val_ratio, n_cpu,
                                                                                  transforms)
    return train_loader, val_loader, test_loader

def get_loader_splitting_val(train_dataset, test_dataset, batch_size_train, batch_size_test, n_cpu, val_ratio):
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True,
                                              num_workers=n_cpu)
    return train_loader, validation_loader, test_loader

def evaluate(model, data_loader, device, accuracy_history=[], loss_history=[], mode="test"):
    model.eval()
    total_samples = len(data_loader.sampler)
    correct_samples = 0
    total_loss = 0
    with no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = log_softmax(model(data), dim=1)
            loss = nll_loss(output, target, reduction='sum')
            _, pred = max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct_samples / total_samples
    if mode == "validation":
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)
    print('\nAverage '+mode+' loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(accuracy) + '%)\n')
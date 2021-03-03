import os
import time
import argparse

import torch
from torch.optim import Adam
from torch.nn.functional import log_softmax, nll_loss
from dataset import CustomDataset, LoadTorchData
from ViT_model import ViT
from matplotlib import pyplot as plt
import numpy as np
torch.manual_seed(42)

parser = argparse.ArgumentParser()

#Data info
parser.add_argument('--dataset_name', type=str, default="CIFAR-10", help='name of the dataset')
parser.add_argument('--load_from_torch', type=bool, default=True, help='define if the dataset must be downloaded using '
                                                                       'torch.datasets')
parser.add_argument('--images_data_path', type=str, default="", help='path of the images contained in dataset (used if load_from_torch = False')
parser.add_argument('--labels_data_path', type=str, default="", help='path of the labels contained in dataset (used if load_from_torch = False')
parser.add_argument('--root_path', type=str, default="./data/datasets/", help='path to download data from pytorch '
                                                                    '(if None the data will not be downloaded)')
parser.add_argument('--download_dataset', type=bool, default=True, help='download dataset from pytorch')
parser.add_argument('--output_graph', type=str, default="./data/graph/", help='path where to save graph')
parser.add_argument('--dump_path', type=str, default="./data/pretrained_models/", help='path of the output files')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes of the datsaset')
parser.add_argument('--n_channels', type=int, default=3, help='number of the color channel of the images in the dataset')
parser.add_argument('--val_ratio', type=int, default=0.2, help='percent of data train to use for validation')

#Training info
parser.add_argument('--n_epochs', type=int, default=200, help='number of the training epochs')
parser.add_argument('--batch_size_train', type=int, default=100, help='batch size to use for the training data')
parser.add_argument('--batch_size_test', type=int, default=100, help='batch size to use for the test data')
parser.add_argument('--lr', type=int, default=0.001, help='learning rate to use for training')
parser.add_argument('--n_cpu', type=int, default=8, help='n cpu to use for data loader')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')

#ViT info
parser.add_argument('--patch_size', type=int, default=7, help='patch size of the ViT model')
parser.add_argument('--image_size', type=int, default=28, help='image size to be input to the ViT model')
parser.add_argument('--dim', type=int, default=64, help='dimension of the ViT model')
parser.add_argument('--n_heads', type=int, default=8, help='number of the heads for the ViT model')
parser.add_argument('--depth', type=int, default=8, help='depth of the ViT model')
parser.add_argument('--feedforward_dim', type=int, default=128, help='dim for the MLP of the ViT model')

opt = parser.parse_args()

def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.sampler)
    model.train()
    sum_losses = 0

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = log_softmax(model(data), dim=1)
        loss = nll_loss(output, target)
        loss.backward()
        optimizer.step()
        sum_losses += loss
        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
    loss_history.append(sum_losses/total_samples)


def evaluate(model, data_loader, loss_history, accuracy_history):
    model.eval()

    total_samples = len(data_loader.sampler)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = log_softmax(model(data), dim=1)
            loss = nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    accuracy_history.append(correct_samples)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

def update_graph(train_loss_history, val_loss_history, val_accuracy_history):
    losses_img_file = os.path.join(opt.output_graph, "training_losses.png")
    acc_img_file = os.path.join(opt.output_graph, "training_accuracy.png")
    epochs = np.arange(1, len(train_loss_history))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(epochs, train_loss_history, label="average train loss")
    plt.plot(epochs, val_loss_history, label="average validation loss")
    plt.legend()
    plt.imsave(losses_img_file)
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.plot(epochs, val_accuracy_history, label="validation accuracy")
    plt.imsave(acc_img_file)

if __name__ == '__main__':
    if opt.load_from_torch:
        train_loader, validation_loader, test_loader = LoadTorchData(root_path=opt.root_path, download=opt.download_dataset).load_dataset(opt.dataset_name,
                                                              opt.batch_size_train, opt.batch_size_test, opt.val_ratio)
    """TO IMPLEMENT: extract train_loader and test_loader (images and labels for each one) of the CustomDataset"""
    dump_path =  os.path.join(opt.dump_path, opt.dataset_name)
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    dump_path = os.path.join(dump_path, "ViT_"+str(opt.patch_size)+str(opt.dim)+".pth")

    start_time = time.time()

    model = ViT(image_size=opt.image_size, patch_size=opt.patch_size, num_classes=opt.n_classes, channels=opt.n_channels,
                dim=opt.dim, depth=opt.depth, num_heads=opt.n_heads, feedforward_dim=opt.feedforward_dim)
    #model = ViT(image_size=opt.image_size, patch_size=opt.patch_size, num_classes=opt.n_classes, dim=opt.dim,
    #            depth=opt.depth, heads=opt.n_heads, mlp_dim=opt.feedforward_dim)
    if opt.cuda:
        model.cuda()

    optimizer = Adam(model.parameters(), lr=opt.lr)

    train_loss_history, val_loss_history, val_accuracy_history = [], [], []
    for epoch in range(1, opt.n_epochs + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history)
        evaluate(model, validation_loader, val_loss_history, val_accuracy_history)
        update_graph(train_loss_history, val_loss_history, val_accuracy_history)
        torch.save(model.state_dict(), dump_path)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
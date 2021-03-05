import os
import time
import argparse
from torch import device, save
from torch.optim import Adam
from torch.nn.functional import log_softmax, nll_loss
from utils import get_ViT_model, get_output_path, get_loader_from_dataset, evaluate
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

#Data info
parser.add_argument('--dataset_name', type=str, default="CIFAR-10", help='name of the dataset')
parser.add_argument('--load_from_torch', type=bool, default=True, help='define if the dataset must be downloaded using '
                                                                       'torch.datasets')

parser.add_argument('--dataset_path', type=str, default="./data/datasets", help='root of the datasets')
parser.add_argument('--output_root_path', type=str, default="./data/", help='path where to save the output results')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes of the dataset')
parser.add_argument('--n_channels', type=int, default=3, help='number of the color channel of the images in the dataset')
parser.add_argument('--val_ratio', type=float, default=0.2, help='percent of data train to use for validation')

#Training info
parser.add_argument('--n_epochs', type=int, default=200, help='number of the training epochs')
parser.add_argument('--batch_size_train', type=int, default=128, help='batch size to use for the training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to use for training')
#parser.add_argument('--lr_decay', type=str, default="cosine", help='learning rate dacay to use (linear or cosine)')
parser.add_argument('--n_cpu', type=int, default=16, help='n cpu to use for data loader')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0", help='cuda device BUS ID')
parser.add_argument('--data_augmentation',  action='store_true', help='use data augmentation')

#ViT info
parser.add_argument('--model_type', type=str, default="Small", help='ViT type (XSmall, Small, Base, Large, Huge)')
parser.add_argument('--hybrid', action='store_true', help='use hybrid ViT')
parser.add_argument('--backbone_name', type=str, default="resnet50", help='backbone name for the hybrit ViT'
                                                                          ' (used if hybrid is setted)')
parser.add_argument('--patch_size', type=int, default=16, help='patch size of the ViT model')
parser.add_argument('--image_size', type=int, default=224, help='image size to be input to the ViT model')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout to use for training')

opt = parser.parse_args()
print(opt)

def train_epoch(model, optimizer, data_loader, loss_history, device):
    total_samples = len(data_loader.sampler)
    model.train()
    sum_losses = 0
    for i, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = log_softmax(model(data), dim=1)
        loss = nll_loss(output, target)
        loss.backward()
        optimizer.step()
        sum_losses += loss.item()
        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
    loss_history.append(sum_losses/len(data_loader))


def update_graph(train_loss_history, val_loss_history, val_accuracy_history, path):
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
    plt.plot(epochs, val_accuracy_history)
    plt.savefig(acc_img_file)

if __name__ == '__main__':
    output_graph_path, dump_file = get_output_path(opt.output_root_path, opt.data_augmentation, opt.dropout,
                                                   opt.model_type, opt.patch_size, opt.dataset_name, opt.hybrid,
                                                   opt.backbone_name)
    train_loader, validation_loader, _ = get_loader_from_dataset(dataset_name=opt.dataset_name,
                                    root_path=opt.dataset_path,
                                    batch_size_train=opt.batch_size_train,
                                    image_size=opt.image_size, augmentation=opt.data_augmentation,
                                    val_ratio=opt.val_ratio, n_cpu=opt.n_cpu)
    start_time = time.time()

    model = get_ViT_model(type=opt.model_type, image_size=opt.image_size, patch_size=opt.patch_size,
                          n_classes=opt.n_classes, n_channels=opt.n_channels, dropout=opt.dropout, hybrid=opt.hybrid,
                           backbone_name=opt.backbone_name)
    device = device("cuda:0" if opt.cuda else "cpu")
    if opt.cuda:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.CUDA_VISIBLE_DEVICES
        model.cuda()
    optimizer = Adam(model.parameters(), lr=opt.lr)
    train_loss_history, val_loss_history, val_accuracy_history = [], [], []
    for epoch in range(1, opt.n_epochs + 1):
        print('Epoch:', epoch)
        train_epoch(model=model, optimizer=optimizer, data_loader=train_loader, loss_history=train_loss_history,
                    device=device)
        evaluate(model=model, data_loader=validation_loader, device=device, loss_history=val_loss_history,
                    accuracy_history=val_accuracy_history, mode="validation")
        update_graph(train_loss_history, val_loss_history, val_accuracy_history, output_graph_path)
        save(model.state_dict(), dump_file)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
import os
import time
import argparse
import sys

from torch import device, save, no_grad
from torch.nn.functional import log_softmax
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler, SGD
from torch import max as torch_max

from utils import get_ViT_model, get_output_path, get_loader_from_dataset, save_result_on_csv, \
    update_graph, get_ViT_name, get_resnet_model

"""
Script that deals with training the models and evaluating their performance
"""

parser = argparse.ArgumentParser()

#Data info
parser.add_argument('--dataset_name', type=str, default="CIFAR-10", help='name of the dataset')
parser.add_argument('--dataset_path', type=str, default="./data/datasets", help='root of the datasets')
parser.add_argument('--output_root_path', type=str, default="./data/", help='path where to save the output results')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes of the dataset')
parser.add_argument('--n_channels', type=int, default=3, help='number of the color channel of the images in the dataset')
parser.add_argument('--val_ratio', type=float, default=0.2, help='percent of data train to use for validation')

#Training info
parser.add_argument('--n_epochs', type=int, default=100, help='number of the training epochs')
parser.add_argument('--batch_size_train', type=int, default=128, help='batch size to use for the training')
parser.add_argument('--batch_size_test', type=int, default=128, help='batch size to use for test')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to use for training')
parser.add_argument('--n_cpu', type=int, default=16, help='n cpu to use for data loader')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0", help='cuda device BUS ID')
parser.add_argument('--data_augmentation',  action='store_true', help='use data augmentation')
parser.add_argument('--eval_type',  type=str, default="both", help='type of evaluation (test, val, both)')
parser.add_argument('--weight_decay',  type=float, default=0., help='weight decay to use')
parser.add_argument('--optimizer',  type=str, default="adam", help='optimizer to use (adam or sgd)')

#Model info
parser.add_argument('--model_type', type=str, default="ViT-XS", help='ViT type (ViT-XS, ViT-S, ViT-B, ViT-L) or'
                                                                     'resnet type (resnet28, resnet34, resnet50, '
                                                                     'resnet101, resnet152')
parser.add_argument('--hybrid', action='store_true', help='use resnet50 hybrid ViT')
parser.add_argument('--patch_size', type=int, default=16, help='patch size of the ViT model')
parser.add_argument('--image_size', type=int, default=224, help='image size to be input to the ViT model')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout to use for training')

opt = parser.parse_args()
print(opt)

def train_epoch(model, optimizer, train_loader, loss_history, acc_history, device, epoch, n_epochs):
    total_samples = len(train_loader.dataset)
    num_batch = len(train_loader)
    model.train()
    sum_losses = 0
    total_correct_samples = 0
    criterion = CrossEntropyLoss()
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = log_softmax(model(data), dim=1)
        loss = criterion(output, target)
        _, pred = torch_max(output, dim=1)
        correct_samples = pred.eq(target).sum()
        accuracy = 100.0 * correct_samples / len(data)
        total_correct_samples += correct_samples
        sum_losses += loss.item()
        loss.backward()
        optimizer.step()
        sys.stdout.write('\rEpoch %03d/%03d [%03d/%03d] -- %s: %.4f -- %s: %.2f --' % (epoch, n_epochs, i+1,
                                                                    num_batch, "train_loss",  loss.item(),
                                                                    "train_acc",  accuracy))
    loss_history.append(sum_losses/num_batch)
    acc_history.append(100.0 * total_correct_samples / total_samples)

def evaluate(model, data_loader, device, acc_history=[], loss_history=[], mode="test", eval_type="both"):
    model.eval()
    total_samples = len(data_loader.sampler)
    num_batch = len(data_loader)
    correct_samples = 0
    total_loss = 0
    criterion = CrossEntropyLoss()
    with no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = log_softmax(model(data), dim=1)
            loss = criterion(output, target)
            _, pred = torch_max(output, dim=1)
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()
    avg_loss = total_loss / num_batch
    accuracy = 100.0 * correct_samples / total_samples
    if mode == "val" or (eval_type == "test" and mode == "test"):
        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        sys.stdout.write(' %s: %.4f -- %s: %.2f \n' % (mode+"_loss",  avg_loss, mode+"_acc",  accuracy))
    return avg_loss, accuracy

if __name__ == '__main__':
    if opt.model_type[:6] == "resnet":
        model_name = opt.model_type
        model = get_resnet_model(resnet_type=opt.model_type, n_classes=opt.n_classes)
    else:
        model_name = get_ViT_name(model_type=opt.model_type, patch_size= opt.patch_size, hybrid=opt.hybrid)
        model = get_ViT_model(type=opt.model_type, image_size=opt.image_size, patch_size=opt.patch_size,
                              n_classes=opt.n_classes, n_channels=opt.n_channels, dropout=opt.dropout,
                              hybrid=opt.hybrid)
    output_graph_path, dump_file = get_output_path(model_name=model_name, root_path=opt.output_root_path,
                                                   dataset_name=opt.dataset_name)
    csv_result_path = os.path.join(opt.output_root_path, "models_results.csv")
    val_ratio = opt.val_ratio
    if opt.eval_type == "test":
        val_ratio = 0
    train_loader, validation_loader, test_loader = get_loader_from_dataset(dataset_name=opt.dataset_name,
                                    root_path=opt.dataset_path,
                                    batch_size_train=opt.batch_size_train, batch_size_test=opt.batch_size_test,
                                    image_size=opt.image_size, augmentation=opt.data_augmentation,
                                    val_ratio=val_ratio, n_cpu=opt.n_cpu)
    device = device("cuda:0" if opt.cuda else "cpu")
    if opt.cuda:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.CUDA_VISIBLE_DEVICES
        model.cuda()
    if opt.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, opt.n_epochs-1)
    train_loss_history, val_loss_history, train_acc_history, val_acc_history = [], [], [], []
    best_test_loss, best_test_acc, best_epoch, best_time = 0.0, 0.0, opt.n_epochs + 1, 0
    start_time = time.time()
    for epoch in range(1, opt.n_epochs + 1):
        train_epoch(model=model, optimizer=optimizer, train_loader=train_loader, loss_history=train_loss_history,
                    acc_history=train_acc_history, device=device, epoch=epoch, n_epochs=opt.n_epochs)
        if opt.eval_type == "val" or opt.eval_type == "both":
            _, _ = evaluate(model=model, data_loader=validation_loader, device=device, loss_history=val_loss_history,
                        acc_history=val_acc_history, mode="val", eval_type=opt.eval_type)
            update_graph(train_loss_history=train_loss_history, val_loss_history=val_loss_history,
                         train_acc_history=train_acc_history, val_acc_history=val_acc_history, path=output_graph_path)
        if opt.eval_type == "test" or opt.eval_type == "both":
            test_loss, test_acc = evaluate(model=model, data_loader=test_loader, device=device, mode="test",
                                           eval_type=opt.eval_type)
            if test_acc > best_test_acc:
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_epoch = epoch
                best_time = time.time() - start_time
        lr_scheduler.step()
        save(model.state_dict(), dump_file)
    exec_time = time.time() - start_time
    if opt.eval_type == "test" or opt.eval_type == "both":
        save_result_on_csv(csv_path=csv_result_path, dataset_name=opt.dataset_name, model_name=model_name,
                           n_epochs=opt.n_epochs, execution_time=exec_time, optimizer=opt.optimizer, dropout=opt.dropout,
                           lr=opt.lr, batch_size=opt.batch_size_train, best_test_loss=best_test_loss,
                           best_test_acc=best_test_acc, best_epoch=best_epoch, best_time=best_time)

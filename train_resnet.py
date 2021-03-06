import argparse
import os
import time

from torch.optim import Adam
from torch import device, save

from utils import get_loader_from_dataset, get_resnet_model, train_epoch, evaluate, update_graph, save_result_on_csv, \
                    get_output_path
parser = argparse.ArgumentParser()

#Data info
parser.add_argument('--dataset_name', type=str, default="CIFAR-10", help='name of the dataset')
parser.add_argument('--dataset_path', type=str, default="./data/datasets", help='root of the datasets')
parser.add_argument('--output_root_path', type=str, default="./data/", help='path where to save the output results')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes of the dataset')
parser.add_argument('--val_ratio', type=float, default=0.2, help='percent of data train to use for validation')
parser.add_argument('--csv_result_path', type=str, default="./data/models_results.csv", help='csv results path')

#Training info
parser.add_argument('--n_epochs', type=int, default=200, help='number of the training epochs')
parser.add_argument('--batch_size_train', type=int, default=128, help='batch size to use for the training')
parser.add_argument('--batch_size_test', type=int, default=128, help='batch size to use for test')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to use for training')
#parser.add_argument('--lr_decay', type=str, default="cosine", help='learning rate dacay to use (linear or cosine)')
parser.add_argument('--n_cpu', type=int, default=16, help='n cpu to use for data loader')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0", help='cuda device BUS ID')
parser.add_argument('--data_augmentation',  action='store_true', help='use data augmentation')

#ViT info
parser.add_argument('--resnet_type', type=str, default="resnet50", help='Type of resnet to train (resnet18, resnet34, resnet50, resnet101, resnet150)')
parser.add_argument('--image_size', type=int, default=224, help='image size to be input to the ViT model')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout to use for training')

opt = parser.parse_args()
print(opt)

if __name__ == '__main__':
    output_graph_path, dump_file = get_output_path(root_path=opt.output_root_path,
                                                   data_augmentation=opt.data_augmentation,
                                                   model_name=opt.resnet_type, dataset_name=opt.dataset_name)
    train_loader, validation_loader, test_loader = get_loader_from_dataset(dataset_name=opt.dataset_name,
                                    root_path=opt.dataset_path,
                                    batch_size_train=opt.batch_size_train, batch_size_test=opt.batch_size_test,
                                    image_size=opt.image_size, augmentation=opt.data_augmentation,
                                    val_ratio=opt.val_ratio, n_cpu=opt.n_cpu)
    model = get_resnet_model(resnet_type=opt.resnet_type, n_classes=opt.n_classes)
    device = device("cuda:0" if opt.cuda else "cpu")
    if opt.cuda:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.CUDA_VISIBLE_DEVICES
        model.cuda()
    optimizer = Adam(model.parameters(), lr=opt.lr)
    train_loss_history, val_loss_history, train_acc_history, val_acc_history = [], [], [], []
    start_time = time.time()
    for epoch in range(1, opt.n_epochs + 1):
        train_epoch(model=model, optimizer=optimizer, train_loader=train_loader, loss_history=train_loss_history,
                    acc_history=train_acc_history, device=device, epoch=epoch, n_epochs=opt.n_epochs)
        evaluate(model=model, data_loader=validation_loader, device=device, loss_history=val_loss_history,
                 acc_history=val_acc_history, mode="validation")
        update_graph(train_loss_history=train_loss_history, val_loss_history=val_loss_history,
                     train_acc_history=train_acc_history, val_acc_history=val_acc_history, path=output_graph_path)
        save(model.state_dict(), dump_file)
    exec_time = time.time() - start_time
    hours, rem = divmod(exec_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Execution training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    print("Evaluating model on data test ... ")
    test_loss, test_acc = evaluate(model, test_loader, device)
    save_result_on_csv(csv_path=opt.csv_result_path, dataset_name=opt.dataset_name, model_name=opt.resnet_type,
                       test_loss=test_loss, n_epochs=opt.n_epochs, test_acc=test_acc, execution_time=exec_time,
                       lr=opt.lr, batch_size=opt.batch_size_train)
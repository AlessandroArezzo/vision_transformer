import argparse
import os

import torch

from utils import get_loader_from_dataset, evaluate, get_ViT_model
parser = argparse.ArgumentParser()

#Data info
parser.add_argument('--model_path', type=str, default="", help='path of the model to evaluate')
parser.add_argument('--dataset_path', type=str, default="./data/datasets", help='root of the datasets')
parser.add_argument('--dataset_name', type=str, default="CIFAR-10", help='name of the dataset to test')
parser.add_argument('--n_channels', type=int, default=3, help='number of the color channel of the images in the dataset')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes of the dataset')

#Test info
parser.add_argument('--batch_size_test', type=int, default=128, help='batch size to use for the test')
parser.add_argument('--n_cpu', type=int, default=8, help='n cpu to use for data loader')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0", help='cuda device BUS ID')

#ViT info
parser.add_argument('--model_type', type=str, default="Small", help='ViT type (XSmall, Small, Base, Large, Huge)')
parser.add_argument('--hybrid', action='store_true', help='use hybrid ViT')
parser.add_argument('--backbone_name', type=str, default="resnet50", help='backbone name for the hybrit ViT'
                                                                          ' (used if hybrid is setted)')
parser.add_argument('--image_size', type=int, default=224, help='image size to be input to the ViT model')
parser.add_argument('--patch_size', type=int, default=16, help='patch size of the ViT model')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout to use for training')
opt = parser.parse_args()
print(opt)

_, _, test_loader = get_loader_from_dataset(dataset_name=opt.dataset_name,
                                             root_path=opt.dataset_path,
                                             batch_size_test=opt.batch_size_test,
                                             image_size=opt.image_size,
                                             n_cpu=opt.n_cpu)

model = get_ViT_model(type=opt.model_type, image_size=opt.image_size, patch_size=opt.patch_size,
                      n_classes=opt.n_classes, n_channels=opt.n_channels, dropout=opt.dropout, hybrid=opt.hybrid,
                      backbone_name=opt.backbone_name)
device = "cpu"
if opt.cuda:
    model.load_state_dict(torch.load(opt.model_path))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.CUDA_VISIBLE_DEVICES
    model.cuda()
    device = "cuda"
try:
    model.load_state_dict(torch.load(opt.model_path))
except RuntimeError:
    if not opt.cuda:
        model.load_state_dict(torch.load(opt.model_path, map_location=torch.device('cpu')))
    else:
        RuntimeError()

evaluate(model, test_loader, device)

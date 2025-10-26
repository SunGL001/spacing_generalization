# run train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
# run train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8
# run train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --cutout --length 20
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pdb
import argparse
import numpy as np
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger
from util.cutout import Cutout, Cutout_input #, Erasing_input
from utils import get_training_dataloader, get_test_dataloader

from model.resnet import ResNet18, ResNet50, ResNet101
from model.wide_resnet import WideResNet

model_options = ['resnet18', 'resnet50', 'resnet101','wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn', 'tiny_imagenet', 'cub200']
method_options = ['StandardDropout','MaxDropout', 'SpacedDropout','MaxSpacedDropout']


parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=80,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--cutout_input', action='store_true', default=False,
                    help='apply cutout_input')    
parser.add_argument('--erasing_input', action='store_true', default=False,
                    help='apply erasing_input')               
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=float, default=16.0,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--run', type=int, default=0,
                    help='running time (default: 0')
parser.add_argument('--drop', type=float, default=0.3,
                    help='drop rate on maxdropout (default: 0.3')
                    
parser.add_argument('--drop_method', type=str, default='MaxDropout', choices=method_options,
                    help='drop method (default: MaxDropout)')
parser.add_argument('--space', type=float, default=10.0,
                    help='space interval on spacedropout ')
parser.add_argument('--gpu_ids', type=str, default='0',
                    help='gpu ids')
parser.add_argument('--wandb_entity', default='sgl0117', type=str, help='downsample size of teacher model')
parser.add_argument('--wandb_project', default='SKD_V2', type=str, help='downsample size of student model')
parser.add_argument('--lr_decay', action='store_true', default=False,
                    help='apply lr decay')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_ids)

args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# print(args)

# Image Preprocessing
# if args.dataset == 'svhn':
#     normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
#                                      std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
# else:
#     normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                      std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

# train_transform = transforms.Compose([])
# if args.data_augmentation:
#     train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
#     train_transform.transforms.append(transforms.RandomHorizontalFlip())
# train_transform.transforms.append(transforms.ToTensor())
# train_transform.transforms.append(normalize)
# train_transform = transforms.Compose([
#     # transforms.Resize(32),
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
#     ])
# if args.cutout:
#     train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))


# # test_transform = transforms.Compose([
# #     transforms.ToTensor(),
# #     normalize])
# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
#     ])

# if args.dataset == 'cifar10':
#     num_classes = 10
#     train_dataset = datasets.CIFAR10(root='/home/liyuan/data/CIFAR',
#                                      train=True,
#                                      transform=train_transform,
#                                      download=True)

#     test_dataset = datasets.CIFAR10(root='/home/liyuan/data/CIFAR',
#                                     train=False,
#                                     transform=test_transform,
#                                     download=True)
# elif args.dataset == 'cifar100':
#     num_classes = 100
#     train_dataset = datasets.CIFAR100(root='/home/liyuan/data/CIFAR',
#                                       train=True,
#                                       transform=train_transform,
#                                       download=True)

#     test_dataset = datasets.CIFAR100(root='/home/liyuan/data/CIFAR',
#                                      train=False,
#                                      transform=test_transform,
#                                      download=True)
# elif args.dataset == 'svhn':
#     num_classes = 10
#     train_dataset = datasets.SVHN(root='/home/liyuan/data/',
#                                   split='train',
#                                   transform=train_transform,
#                                   download=True)

#     extra_dataset = datasets.SVHN(root='/home/liyuan/data/',
#                                   split='extra',
#                                   transform=train_transform,
#                                   download=True)

#     # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
#     data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
#     labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
#     train_dataset.data = data
#     train_dataset.labels = labels

#     test_dataset = datasets.SVHN(root='/home/liyuan/data/',
#                                  split='test',
#                                  transform=test_transform,
#                                  download=True)

# # Data Loader (Input Pipeline)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=args.batch_size,
#                                            shuffle=True,
#                                            pin_memory=True,
#                                            num_workers=2)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=args.batch_size,
#                                           shuffle=False,
#                                           pin_memory=True,
#                                           num_workers=2)


if args.dataset == 'imagenet':
    num_classes = 1000
elif args.dataset == 'imagenet_100':
    num_classes = 100
elif args.dataset == 'cifar100':
    num_classes = 100
elif args.dataset == 'tiny_imagenet':
    num_classes = 200
elif args.dataset == 'cub200':
    num_classes = 200


#data preprocessing:
train_loader = get_training_dataloader(
    dataset=args.dataset,
    num_workers=4,
    batch_size=args.batch_size,
    shuffle=True)

test_loader = get_test_dataloader(
    dataset=args.dataset,
    num_workers=4,
    batch_size=args.batch_size,
    shuffle=True)
length = int(len(train_loader))

if args.model == 'resnet18':
    if args.drop_method == 'MaxDropout':
        cnn = ResNet18(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes)
    elif args.drop_method == 'StandardDropout':
        cnn = ResNet18(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes)
    elif args.drop_method == 'SpacedDropout':
        cnn = ResNet18(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes, space=int(args.space*len(train_loader)))
    elif args.drop_method == 'MaxSpacedDropout':
        cnn = ResNet18(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes, space=int(args.space*len(train_loader)))
elif args.model == 'resnet50':
    if args.drop_method == 'MaxDropout':
        cnn = ResNet50(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes)
    elif args.drop_method == 'StandardDropout':
        cnn = ResNet50(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes)
    elif args.drop_method == 'SpacedDropout':
        cnn = ResNet50(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes, space=int(args.space*len(train_loader)))
    elif args.drop_method == 'MaxSpacedDropout':
        cnn = ResNet50(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes, space=int(args.space*len(train_loader)))
elif args.model == 'resnet101':
    if args.drop_method == 'MaxDropout':
        cnn = ResNet101(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes)
    elif args.drop_method == 'StandardDropout':
        cnn = ResNet101(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes)
    elif args.drop_method == 'SpacedDropout':
        cnn = ResNet101(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes, space=int(args.space*len(train_loader)))
    elif args.drop_method == 'MaxSpacedDropout':
        cnn = ResNet101(drop_method=args.drop_method, drop=args.drop, num_classes=num_classes, space=int(args.space*len(train_loader)))  
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=4, dropRate=args.drop)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=args.drop)

cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

if args.lr_decay:
    if args.dataset == 'svhn':
        scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
    else:
        scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

test_id = args.dataset + '_' + args.model +'_spaced_'+str(args.space)+'_'+str(args.drop_method)  +'_'+str(args.drop)  # +'_'+str(args.learning_rate)
if args.lr_decay:
    test_id = test_id + '_decay'
elif args.cutout_input:
    test_id = 'Cutout_' + str(args.drop_method) + '_space_' + str(args.space) + '_length_' + str(args.length)
elif args.erasing_input:
    test_id = 'Erasing_' + str(args.drop_method) + '_space_' + str(args.space) + '_length_' + str(args.length)
elif args.cutout:
    test_id = 'Cutout_standard_' + str(args.drop_method)  + '_length_' + str(args.length)

# filename = 'logs/' + test_id + '.csv'
# csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


#so the only way is to create a new wandb log
wandb.init(dir='/data/sunguanglong/results_spcaed_distillation/wandb_SKD',entity=args.wandb_entity, project=args.wandb_project, name=test_id, config=args)
wandb_url = wandb.run.get_url()
print(f"Wandb URL: {wandb_url}")


def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc

best_acc=0.0
count=0
for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    # progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        # progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        if args.cutout_input or args.erasing_input:
            if args.space == 0:
                t_lenth=args.length
            elif args.space > 500:
                t_lenth = args.length * np.random.rand()
            else:
                t_lenth = args.length * np.abs(np.sin((np.pi/(args.space*length))*count))

            if args.cutout_input:
                images = Cutout_input(images, n_holes=args.n_holes, length=int(t_lenth))
            elif args.erasing_input:
                images = Erasing_input(img=images, p=1.0, sl=t_lenth, sh=t_lenth, r1=0.3)

        count += 1
        #cnn.zero_grad()
        cnn_optimizer.zero_grad()
        pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        # progress_bar.set_postfix(
        #     xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
        #     acc='%.5f' % accuracy)
        wandb.log({'train_loss': xentropy_loss_avg / (i + 1)}, commit=True)

    test_acc = test(test_loader)
    # tqdm.write('test_acc: %.5f' % (test_acc))
    wandb.log({'test_acc': test_acc}, commit=True)

    # save
    if best_acc < test_acc:
        weights_path = os.path.join('/data/sunguanglong/results_spcaed_distillation/fig3_dropout', test_id + '.pkl')
        print('saving weights file to {}'.format(weights_path))
        torch.save({'model': cnn.state_dict()}, weights_path)
        best_acc = test_acc
        print('Epoch:', epoch, 'best_acc: %.4f' % (best_acc))
        continue

    wandb.log({'best_acc': best_acc}, commit=True)


    # scheduler.step(epoch)  # Use this line for PyTorch <1.4
    if args.lr_decay:
        scheduler.step()     # Use this line for PyTorch >=1.4

    # row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    # csv_logger.writerow(row)

# torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
# csv_logger.close()
wandb.finish()

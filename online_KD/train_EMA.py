# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import wandb
import copy

from models.resnet import resnet18, resnet50, resnet101

from conf import settings
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights


@torch.no_grad()
def test(model, loader):
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            _, pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    return val_acc


class EMA:
    def __init__(self, model, decay):
        # Initialize EMA with a deepcopy of the original model
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        for param in self.ema_model.parameters():
            param.detach_()
    
    def update(self, model):
        # Update EMA model parameters based on the original model
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data[:] = ema_param.data[:] * self.decay + param.data[:] * (1 - self.decay)
    
    def state_dict(self):
        # Return the EMA model's state dictionary
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict):
        # Load EMA model state dictionary
        self.ema_model.load_state_dict(state_dict)
    
    def to(self, device):
        # Move EMA model to the specified device (CPU or GPU)
        self.ema_model = self.ema_model.to(device)

    def __call__(self, x):
        # Call EMA model for inference
        return self.ema_model(x)



# class EMA:
#     def __init__(self, model, model_ema, optimizer, args):
#         self.model = model
#         self.model_ema = model_ema
#         self.optimizer = optimizer
#         self.interval = args.interval
#         self.max_epochs = args.epoch
#         self.decay = args.alpha
#         self.batch_size = args.b
#         self.criterion=nn.CrossEntropyLoss()
#         self.shadow = {}

#     @torch.no_grad()
#     def update_ema_parameters(self):
#         """
#         Momentum update of the key encoder
#         """
#         # for param, param_ema in zip(self.model.parameters(), self.model_ema.parameters()):
#         #     param_ema.data = param_ema.data * self.decay + param.data * (1. - self.decay)
#         # self.model_ema.load_state_dict(self.model.state_dict())
#         for name, param in self.model.named_parameters():
#             new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
#             self.shadow[name] = new_average.clone()
    
#     @torch.no_grad()
#     def apply_shadow(self):
#         for name, param in self.model_ema.named_parameters():
#             param.data = self.shadow[name]

#     @torch.no_grad()
#     def initializes_ema_network(self):
#         # init momentum network as encoder net
#         for name, param in self.model.named_parameters():
#             self.shadow[name] = param.data.clone()
#         # for param, param_ema in zip(self.model.parameters(), self.model_ema.parameters()):
#         #     param_ema.data.copy_(param.data)  # initialize
#         #     param_ema.requires_grad = False  # not update by gradient
    
#     @torch.no_grad()
#     def test(self, model, loader):
#         original_mode = model.training
#         model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
#         correct = 0.
#         total = 0.
#         for images, labels in loader:
#             images = images.cuda()
#             labels = labels.cuda()

#             with torch.no_grad():
#                 _, pred = model(images)

#             pred = torch.max(pred.data, 1)[1]
#             total += labels.size(0)
#             correct += (pred == labels).sum().item()

#         val_acc = correct / total
#         model.train(original_mode)
#         return val_acc

#     def train(self, training_loader, test_loader, test_id):

#         self.initializes_ema_network()
#         best_acc=0.0
#         count=0
#         for epoch in range(self.max_epochs):
#             # loss_avg = 0.
#             correct = 0.
#             total = 0.

#             for images, labels in training_loader:

#                 count += 1
#                 images = images.cuda()
#                 labels = labels.cuda()

#                 #cnn.zero_grad()
#                 self.optimizer.zero_grad()
#                 _, pred = self.model(images)

#                 loss = self.criterion(pred, labels)
#                 loss.backward()
#                 self.optimizer.step()

#                 # loss_avg += loss.item()

#                 # Calculate running average of accuracy
#                 pred = torch.max(pred.data, 1)[1]
#                 total += labels.size(0)
#                 correct += (pred == labels.data).sum().item()
#                 accuracy = correct / total
#                 wandb.log({'train_loss': loss.item()}, commit=True)

#                 self.update_ema_parameters()  # update the ema model

#             self.apply_shadow()
#             acc_model_ema = self.test(self.model_ema, test_loader)
#             acc_model = self.test(self.model, test_loader)
#             self.model_ema.eval()

#             print('Epoch:', epoch, 'acc_model_ema: %.5f' % (acc_model_ema))
#             print('Epoch:', epoch, 'acc_model %.5f' % (acc_model))


#             # save
#             if best_acc < acc_model_ema:
#                 weights_path = os.path.join('/home/liyuan/data/fig4_ema', test_id + '.pkl')
#                 print('saving weights file to {}'.format(weights_path))
#                 torch.save({'model': self.model_ema.state_dict()}, weights_path)
#                 best_acc = acc_model_ema
#                 continue
#             wandb.log({'acc_model_ema': acc_model_ema}, commit=True)
#             wandb.log({'best_acc': best_acc}, commit=True)

#             # scheduler.step(epoch)  # Use this line for PyTorch <1.4
#             # scheduler.step()     # Use this line for PyTorch >=1.4



# def initializes_ema_network(model, model_ema):
#     # init momentum network as encoder net
#     for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
#         param_k.data.copy_(param_q.data)  # initialize
#         param_k.requires_grad = False  # not update by gradient


# Function to update EMA model
# def update_ema(ema_model, model, ema_decay):
#     with torch.no_grad():
#         for ema_param, param in zip(ema_model.parameters(), model.parameters()):
#             ema_param.data = ema_param.data * ema_decay + param.data * (1 - ema_decay)

# def update_ema_network(model, ema_model, ema_decay):
#     with torch.no_grad():
#         for ema_param, param in zip(ema_model.parameters(), model.parameters()):
#             ema_param.data[:] = ema_param[:].data[:] * ema_decay + (1 - ema_decay) * param[:].data[:]
#         return ema_model
#             # ema_param.data.copy_(param.data)

# @torch.no_grad()
# def update_ema_network(model, model_ema, alpha=0.9):
#     """
#     Momentum update of the key encoder
#     """
#     # for param, param_ema in zip(model.state_dict().values(), model_ema.state_dict().values()):
#     #     param_ema.data = param_ema.data * alpha + param.data * (1. - alpha)
#     # model_ema.eval()
#     # # model_ema.load_state_dict(model.state_dict())
#     for n, param_ema in model_ema.named_parameters():
#         param = model.state_dict()[n]
#         param_ema.data = param_ema.data * alpha + param.data * (1. - alpha)
#         param_ema.requires_grad = False
#     model_ema.eval()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, default='spaced_distillation', required=True, help='net type')
    parser.add_argument('-net', type=str, default='resnet18', required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-lr_decay', action='store_true', default=False, help='use learning rate decay or not')
    parser.add_argument('-epoch', type=int, default=80, help='total training epoch')
    parser.add_argument('-alpha', type=float, default=0.999, help='distillation student loss weight')
    parser.add_argument('-interval', type=float, default=1, help='interval epoch for updating student')
    parser.add_argument('-downsample_factor', default='1.0', type=float, help='downsample size of model')
    parser.add_argument('-wandb_entity', default='sgl0117', type=str, help='downsample size of teacher model')
    parser.add_argument('-wandb_project', default='SKD_V2', type=str, help='downsample size of student model')
    parser.add_argument('-dataset', default='cifar100', type=str, help='cifar100, Imagenet')
    parser.add_argument('-gpu_ids', default='0', type=str, help='gpu ids for training')
    parser.add_argument('-optimizer', default='SGD', type=str, help='optimizer')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_ids)
    device_ids = [i for i in range(torch.cuda.device_count())]

    # net = get_network(args)

    if args.lr_decay:
        if args.dataset == 'imagenet':
            num_classes = 1000
            milestones = settings.MILESTONES_IMAGENET
            epoch_total = settings.EPOCH_IMAGENET
        elif args.dataset == 'cifar100':
            num_classes = 100
            milestones = settings.MILESTONES
            epoch_total = settings.EPOCH
        elif args.dataset == 'imagenet_100':
            num_classes = 100
            milestones = settings.MILESTONES
            epoch_total = settings.EPOCH
        elif args.dataset == 'tiny_imagenet':
            num_classes = 200
            milestones = settings.MILESTONES
            epoch_total = settings.EPOCH
    else:
        if args.dataset == 'imagenet':
            num_classes = 1000
            epoch_total = args.epoch
        elif args.dataset == 'imagenet_100':
            num_classes = 100
            epoch_total = args.epoch
        elif args.dataset == 'cifar100':
            num_classes = 100
            epoch_total = args.epoch
        elif args.dataset == 'tiny_imagenet':
            num_classes = 200
            epoch_total = args.epoch


    #data preprocessing:
    training_loader = get_training_dataloader(
        dataset=args.dataset,
        num_workers=4,
        batch_size=args.b,
        shuffle=True)

    test_loader = get_test_dataloader(
        dataset=args.dataset,
        num_workers=4,
        batch_size=args.b,
        shuffle=True)
    
    iter_per_epoch = len(training_loader)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.net == 'resnet18':
        model = resnet18(downsample_factor=1, num_classes=num_classes).cuda()
    elif args.net == 'resnet50':
        model = resnet50(downsample_factor=1, num_classes=num_classes).cuda()
    elif args.net == 'resnet101':
        model = resnet101(downsample_factor=1, num_classes=num_classes).cuda()

    ema = EMA(model, decay=args.alpha)
    # model_ema.load_state_dict(model.state_dict())
    # for param in model_ema.parameters():
    #     param.requires_grad = False


    # if torch.cuda.device_count() > 0:
    #     model=nn.DataParallel(model, device_ids=device_ids)
    
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.lr_decay:
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2) #learning rate decay
        # warmup_scheduler_teacher = WarmUpLR(optimizer_teacher, iter_per_epoch * args.warm)

    test_id = 'ema_' + args.dataset + '_' + args.net +'_spaced_'+str(args.interval)+'_alpha_'+str(args.alpha)
    # checkpoint_path = os.path.join('/data/sunguanglong/results_spcaed_distillation/checkpoints', args.net_teacher + '_' + args.net_student+ '_' + args.dataset,
    #         'interval'+str(args.interval_rate) + '_temp' + str(args.temp) + '_alpha' + str(args.alpha) + '_batchsize' + str(args.b) + '_' + str(args.feature_loss), settings.TIME_NOW)

    #so the only way is to create a new wandb log
    # wandb.init(dir='/data/sunguanglong/results_spcaed_distillation/wandb_SKD',entity=args.wandb_entity, project=args.wandb_project, name=args.net_teacher+'_'+args.net_student+'_interval'+str(args.interval_rate) + 
    #            '_dp' + str(args.downsample_factor) + '_temp' + str(args.temp) + '_alpha' + str(args.alpha) + '_batchsize' + str(args.b) + '_' + str(args.feature_loss) + '_' + str(args.dataset), config=args)
    wandb.init(dir='/data/sunguanglong/results_spcaed_distillation/wandb_SKD',entity=args.wandb_entity, project=args.wandb_project, name=test_id, config=args)
    wandb_url = wandb.run.get_url()
    print(f"Wandb URL: {wandb_url}")


    best_acc=0.0
    count=0
    if args.interval == 0:
        t_interval = 1
    elif args.interval <= 500:
        t_interval = int(args.interval*iter_per_epoch)
    else:
        t_interval = int(0.4*iter_per_epoch * np.random.rand())+1

    for epoch in range(args.epoch):
        # loss_avg = 0.
        correct = 0.
        total = 0.

        for images, labels in training_loader:

            count += 1
            images = images.cuda()
            labels = labels.cuda()

            #cnn.zero_grad()
            optimizer.zero_grad()
            _, pred = model(images)

            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            # loss_avg += loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total
            wandb.log({'train_loss': loss.item()}, commit=True)

           
            if count==t_interval:
                ema.update(model) 
                count=0
                if args.interval > 500:
                    t_interval = int(0.4*iter_per_epoch * np.random.rand())+1

        acc_model_ema = test(ema, test_loader)
        acc_model = test(model, test_loader)

        print('Epoch:', epoch, 'acc_model_ema: %.5f' % (acc_model_ema))
        print('Epoch:', epoch, 'acc_model %.5f' % (acc_model))
        wandb.log({'acc_model': acc_model}, commit=True)
        wandb.log({'acc_model_ema': acc_model_ema}, commit=True)

        # save
        if best_acc < acc_model_ema:
            weights_path = os.path.join('/data/sunguanglong/results_spcaed_distillation/fig4_ema', test_id + '.pkl')
            print('saving weights file to {}'.format(weights_path))
            torch.save({'model': model.state_dict()}, weights_path)
            best_acc = acc_model_ema
            continue

        wandb.log({'best_acc': best_acc}, commit=True)

        # scheduler.step(epoch)  # Use this line for PyTorch <1.4
        # scheduler.step()     # Use this line for PyTorch >=1.4


    wandb.finish()

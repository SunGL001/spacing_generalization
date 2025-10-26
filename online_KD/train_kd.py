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

from models.nets import teacher_net, student_net

from conf import settings
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights


def train(teacher, student, mod, epoch, temp, alpha, beta, interval=0):
    ## mod: a parameter to control which part of the network to train, 'teacher', 'student' or 'both'
    
    start = time.time()

    if mod == 'both':
        global buffer
        teacher.train()
        student.train() 
    elif mod == 'teacher':
        teacher.train()
    elif mod =='student':
        teacher.eval()
        student.train()

    for batch_index, (images, labels) in enumerate(training_loader):

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        if mod == 'both':
            buffer['images'].append(images.cpu())
            buffer['labels'].append(labels.cpu())
            optimizer_teacher.zero_grad()
            feature_teacher, pred_teacher = teacher(images)
            loss_teacher = hard_loss(pred_teacher, labels)
            loss_teacher.backward()
            optimizer_teacher.step()
        
        elif mod == 'student':
            with torch.no_grad():
                feature_teacher, pred_teacher = teacher(images)

            optimizer_student.zero_grad()
            feature_student, pred_student = student(images)
            loss_hard=hard_loss(pred_student,labels)
            loss_soft = soft_loss(F.softmax(pred_student/temp, dim=1),
            F.softmax(pred_teacher.detach()/temp, dim=1))
            loss_feature=feature_loss_function(feature_student, feature_teacher.detach())
            loss_student = (1-alpha)*loss_hard + alpha*loss_soft + beta*loss_feature        
            loss_student.backward()
            optimizer_student.step()
            wandb.log({'Train/student': loss_student.item(), 'Train/soft_loss': loss_soft.item(), 
                        'Train/hard_loss': loss_hard.item(), 'Train/feature_loss': loss_feature.item()}, commit=True)
            
        elif mod == 'teacher':
            optimizer_teacher.zero_grad()
            feature_teacher, pred_teacher = teacher(images)
            loss_teacher = hard_loss(pred_teacher, labels)
            loss_teacher.backward()
            optimizer_teacher.step()

        # decide whether to space distillation
        if mod != 'both':
            space = False
        elif interval == 0:
            space = True
        elif n_iter % interval == 0:
            space = True
        else:
            space = False

        if space:
            for t_images, t_labels in zip(buffer['images'], buffer['labels']):

                if args.gpu:
                    t_labels = t_labels.cuda()
                    t_images = t_images.cuda()
         
                # t_iter=epoch*len(training_loader) + t_batch_idx + 1
                if interval == 0:
                    feature_teacher, pred_teacher = feature_teacher.detach(), pred_teacher.detach()
                else:
                    with torch.no_grad():
                        feature_teacher, pred_teacher = teacher(t_images)

                optimizer_student.zero_grad()
                feature_student, pred_student = student(t_images)
                loss_hard=hard_loss(pred_student,t_labels)
                loss_soft = soft_loss(F.softmax(pred_student/temp, dim=1),
                F.softmax(pred_teacher.detach()/temp, dim=1))
                loss_feature=feature_loss_function(feature_student, feature_teacher.detach())
                loss_student = (1-alpha)*loss_hard + alpha*loss_soft + beta*loss_feature
                
                loss_student.backward()
                optimizer_student.step()

                wandb.log({'Train/student': loss_student.item(), 'Train/soft_loss': loss_soft.item(), 
                           'Train/hard_loss': loss_hard.item(), 'Train/feature_loss': loss_feature.item()}, commit=False)

            buffer={'images':[], 'labels':[]}
            # interval += int(len(trainloader)*random.uniform(0.5,1.0))
 

        if space:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss_teacher.item(),
                optimizer_teacher.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(training_loader.dataset)
            ))

        #update training loss for each iteration
        if mod != 'student':
            wandb.log({'Train/teacher': loss_teacher.item()}, commit=True)

        # if epoch <= args.warm:
        #     warmup_scheduler_teacher.step()
        #     warmup_scheduler_student.step()

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(teacher, student, epoch=0, wb=True):
    global best_acc
    start = time.time()
    teacher.eval()
    student.eval()

    test_loss = 0.0 # cost function error
    correct_teacher = 0
    correct_student = 0

    for (images, labels) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        _, output_teacher = teacher(images)
        # loss = hard_loss(output_teacher, labels)

        _, output_student = student(images)

        loss = hard_loss(output_student, labels)
        

        test_loss += loss.item()
        
        _, preds_teacher = output_teacher.max(1)
        correct_teacher += preds_teacher.eq(labels).sum()
        _, preds_student = output_student.max(1)
        correct_student += preds_student.eq(labels).sum()

    finish = time.time()
    if best_acc < correct_student.float() / len(test_loader.dataset):
        best_acc = correct_student.float() / len(test_loader.dataset)
        print("Best Accuracy Updated: ", best_acc * 100)

    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy_teacher: {:.4f}, Accuracy_student: {:.4f}, Accuracy_diff:{:.4f}'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        correct_teacher.float() / len(test_loader.dataset),
        correct_student.float() / len(test_loader.dataset),
        (correct_student.float()-correct_teacher.float()) / len(test_loader.dataset)
    ))
    
    #add informations to wandb
    if wb:
    #     wandb.log({'Test/Average_loss/': test_loss / len(test_loader.dataset),
    #                 'Test/Accuracy_teacher': correct_teacher.float() / len(test_loader.dataset)}, commit=True)

    # return correct_teacher.float() / len(test_loader.dataset)
        wandb.log({'Test/Average_loss/': test_loss / len(test_loader.dataset),
                    'Test/Accuracy_teacher': correct_teacher.float() / len(test_loader.dataset),
                    'Test/Accuracy_student': correct_student.float() / len(test_loader.dataset),
                    'Test/Accuracy_diff': (correct_student.float() - correct_teacher.float()) / len(test_loader.dataset)}, commit=True)
        wandb.log({'Test/best_acc': best_acc}, commit=True)

    return correct_teacher.float() / len(test_loader.dataset), correct_student.float() / len(test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-net', type=str, default='spaced_distillation', required=True, help='net type')
    parser.add_argument('-net_teacher', type=str, default='resnet18', required=True, help='net type')
    parser.add_argument('-net_student', type=str, default='resnet18', required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-lr_decay', action='store_true', default=False, help='use learning rate decay or not')
    parser.add_argument('-epoch', type=int, default=80, help='total training epoch')
    parser.add_argument('-temp', type=float, default=3.0, help='kl divergence temperature')
    parser.add_argument('-alpha', type=float, default=0.3, help='distillation student loss weight')
    parser.add_argument('-beta', type=float, default=0.03, help='feature loss weight')
    parser.add_argument('-interval_rate', type=float, default=1.0, help='interval epoch rate for updating student')
    parser.add_argument('-downsample_factor', default='1.0', type=float, help='downsample size of model')
    parser.add_argument('-wandb_entity', default='none', type=str, help='downsample size of teacher model')
    parser.add_argument('-wandb_project', default='spaced_distillation', type=str, help='downsample size of student model')
    parser.add_argument('-feature_loss', default='dist', type=str, help='loss function for feature distillation')
    parser.add_argument('-dataset', default='cifar100', type=str, help='cifar100, Imagenet')
    parser.add_argument('-gpu_ids', default='0,1', type=str, help='gpu ids for training')
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
        elif args.dataset == 'tiny_imagenet':
            num_classes = 200
            milestones = settings.MILESTONES
            epoch_total = settings.EPOCH
    else:
        if args.dataset == 'imagenet':
            num_classes = 1000
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

    hard_loss = nn.CrossEntropyLoss()
    soft_loss = nn.KLDivLoss(reduction="batchmean")
    if args.feature_loss == 'L1':
        feature_loss_function = nn.L1Loss(reduction='mean')
    elif args.feature_loss == 'L2':
        feature_loss_function = nn.MSELoss(reduction='mean')
    elif args.feature_loss == 'CE':
        feature_loss_function = nn.CrossEntropyLoss()
    elif args.feature_loss == 'SmoothL1':
        feature_loss_function = nn.SmoothL1Loss(reduction='mean')
    elif args.feature_loss == 'dist':
        feature_loss_function = torch.dist

    net_teacher = teacher_net(args=args, num_classes=num_classes).cuda()
    net_student = student_net(args=args, num_classes=num_classes, teacher_feature_size=net_teacher.feature_size).cuda()

    if torch.cuda.device_count() > 0:
        net_teacher=nn.DataParallel(net_teacher, device_ids=device_ids)
        net_student=nn.DataParallel(net_student, device_ids=device_ids)
    
    if args.optimizer == 'Adam':
        optimizer_teacher = optim.Adam(net_teacher.parameters(), lr=args.lr, weight_decay=5e-4)
        optimizer_student = optim.Adam(net_student.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer == 'SGD':
        optimizer_teacher = optim.SGD(net_teacher.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer_student = optim.SGD(net_student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.lr_decay:
        train_scheduler_teacher = optim.lr_scheduler.MultiStepLR(optimizer_teacher, milestones=milestones, gamma=0.2) #learning rate decay
        # warmup_scheduler_teacher = WarmUpLR(optimizer_teacher, iter_per_epoch * args.warm)
        train_scheduler_student = optim.lr_scheduler.MultiStepLR(optimizer_student, milestones=milestones, gamma=0.2) #learning rate decay
        # warmup_scheduler_student = WarmUpLR(optimizer_student, iter_per_epoch * args.warm)


    checkpoint_path = os.path.join('./checkpoints', args.net_teacher + '_' + args.net_student+ '_' + args.dataset,
            'interval'+str(args.interval_rate) + '_dp' + str(args.downsample_factor) + '_temp' + str(args.temp) + '_alpha' + str(args.alpha) + '_batchsize' + str(args.b) + '_' + str(args.feature_loss), settings.TIME_NOW)

    #so the only way is to create a new wandb log
    # wandb.init(dir='./wandb_SKD',entity=args.wandb_entity, project=args.wandb_project, name=args.net_teacher+'_'+args.net_student+'_interval'+str(args.interval_rate) + 
    #            '_dp' + str(args.downsample_factor) + '_temp' + str(args.temp) + '_alpha' + str(args.alpha) + '_batchsize' + str(args.b) + '_' + str(args.feature_loss) + '_' + str(args.dataset), config=args)
    wandb.init(dir='./wandb_SKD',entity=args.wandb_entity, project=args.wandb_project, name=args.net_teacher+'_'+args.net_student+'_interval'+str(args.interval_rate) + 
            '_lr' + str(args.lr) + '_temp' + str(args.temp) + '_alpha' + str(args.alpha) + '_batchsize' + str(args.b) + '_' + str(args.feature_loss) + '_' + str(args.dataset), config=args)

    wandb_url = wandb.run.get_url()
    print(f"Wandb URL: {wandb_url}")

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    
    # if args.resume:
    #     best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net_student, recent_folder))
    #     if best_weights:
    #         weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net_student, recent_folder, best_weights)
    #         print('found best acc weights file:{}'.format(weights_path))
    #         print('load best training file to test acc...')
    #         net_student.load_state_dict(torch.load(weights_path))
    #         best_acc = eval_training(wb=False)
    #         print('best acc is {:0.2f}'.format(best_acc))

    #     recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net_student, recent_folder))
    #     if not recent_weights_file:
    #         raise Exception('no recent weights file were found')
    #     weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net_student, recent_folder, recent_weights_file)
    #     print('loading weights file {} to resume training.....'.format(weights_path))
    #     net_student.load_state_dict(torch.load(weights_path))
    #     resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net_student, recent_folder))


    best_acc = 0.0
    if args.interval_rate < 50:
        mod = 'both'
        interval = int(len(training_loader)*args.interval_rate)
        buffer={'images':[], 'labels':[]} 
        for epoch in range(1, epoch_total + 1):
            # if epoch > args.warm:
            #     train_scheduler_teacher.step(epoch)
            #     train_scheduler_student.step(epoch)
            # if args.resume:
            #     if epoch <= resume_epoch:
            #         continue
            train(teacher=net_teacher, student=net_student, mod=mod, epoch=epoch, temp=args.temp, alpha=args.alpha, beta=args.beta, interval=interval)
            acc_teacher, acc_student = eval_training(teacher=net_teacher, student=net_student, epoch=epoch, wb=True)  

    else:
        # train teacher first
        mod = 'teacher'
        for epoch in range(1, epoch_total + 1):
            train(teacher=net_teacher, student=net_student, mod=mod, epoch=epoch, temp=args.temp, alpha=args.alpha, beta=args.beta)
        # then train student
        mod = 'student'
        for epoch in range(1, epoch_total + 1):
            train(teacher=net_teacher, student=net_student, mod=mod, epoch=epoch, temp=args.temp, alpha=args.alpha, beta=args.beta)               
            acc_teacher, acc_student = eval_training(teacher=net_teacher, student=net_student, epoch=epoch, wb=True)
    

        # acc_student = eval_training(epoch,wb=True)

        #start to save best performance model after learning rate decay to 0.01
        # if epoch > milestones[1] and best_acc < acc_student:
        #     weights_path = checkpoint_path.format(net=args.net_student, epoch=epoch, type='best')
        #     print('saving weights file to {}'.format(weights_path))
        #     torch.save({'teacher_model': net_teacher.module.state_dict(),
        #                 'student_model': net_student.module.state_dict(),}, weights_path)
        #     best_acc = acc_student
        #     continue

        # if not epoch % settings.SAVE_EPOCH:
        #     weights_path = checkpoint_path.format(net=args.net_student, epoch=epoch, type='regular')
        #     print('saving weights file to {}'.format(weights_path))
        #     torch.save({'teacher_model': net_teacher.module.state_dict(),
        #                 'student_model': net_student.module.state_dict(),}, weights_path)

    wandb.finish()

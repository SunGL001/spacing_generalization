import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import *
import torch.nn.functional as F
from autoaugment import CIFAR10Policy
from cutout import Cutout
import wandb

parser = argparse.ArgumentParser(description='Self-Distillation CIFAR Training')
parser.add_argument('--model', default="resnet18", type=str, help="resnet18|resnet34|resnet50|resnet101|resnet152|"
                                                                   "wideresnet50|wideresnet101|resnext50|resnext101")
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar100|cifar10")
# parser.add_argument('--epoch', default=250, type=int, help="training epochs")
parser.add_argument('--epoch', default=80, type=int, help="training epochs")
parser.add_argument('--loss_coefficient', default=0.3, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
parser.add_argument('--dataset_path', default="data", type=str)
parser.add_argument('--autoaugment', default=True, type=bool)
parser.add_argument('--temperature', default=3.0, type=float)
parser.add_argument('--batchsize', default=128, type=int)
# parser.add_argument('--init_lr', default=0.1, type=float)
parser.add_argument('--init_lr', default=0.01, type=float)
parser.add_argument('--wandb_entity', default='none', type=str, help='wandb_entity')
parser.add_argument('--wandb_project', default='SKD', type=str, help='wandb_project')
parser.add_argument('--interval_rate', default=1.0, type=float)
parser.add_argument('--gpu_ids', default='0,1', type=str, help='gpu ids for training')
args = parser.parse_args()
print(args)

import os
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_ids)
device_ids = [i for i in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/args.temperature, dim=1)
    softmax_targets = F.softmax(targets/args.temperature, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


if args.autoaugment:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                             transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
                             Cutout(n_holes=1, length=16),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
else:
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(
        root=args.dataset_path,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=args.dataset_path,
        train=False,
        download=True,
        transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4
    )
elif args.dataset == 'tiny_imagenet':
    from utils import load_tinyimagenet
    id_dic = {}
    root = '/data/datasets/tiny-imagenet-200'
    for i, line in enumerate(open(root+'/wnids.txt','r')):
        id_dic[line.replace('\n', '')] = i
    # num_classes = len(id_dic)
    trainloader = load_tinyimagenet(root=root, batch_size=args.batchsize, num_workers=4, split='train', shuffle=True, id_dic=id_dic)
    testloader = load_tinyimagenet(root=root, batch_size=args.batchsize, num_workers=4, split='val', shuffle=True, id_dic=id_dic)

elif args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(
        root=args.dataset_path,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.dataset_path,
        train=False,
        download=True,
        transform=transform_test
    )

if args.dataset == "cifar100":
    if args.model == "resnet18":
        net = resnet18()
    if args.model == "resnet34":
        net = resnet34()
    if args.model == "resnet50":
        net = resnet50()
    if args.model == "resnet101":
        net = resnet101()
    if args.model == "resnet152":
        net = resnet152()
    if args.model == "wideresnet50":
        net = wide_resnet50_2()
    if args.model == "wideresnet101":
        net = wide_resnet101_2()
    if args.model == "resnext50_32x4d":
        net = resnet18()
    if args.model == "resnext101_32x8d":
        net = resnext101_32x8d()
elif args.dataset == "tiny_imagenet":
    if args.model == "resnet18":
        net = resnet18(num_classes=200)
    if args.model == "resnet34":
        net = resnet34(num_classes=200)
    if args.model == "resnet50":
        net = resnet50(num_classes=200)
    if args.model == "resnet101":
        net = resnet101(num_classes=200)
    if args.model == "resnet152":
        net = resnet152(num_classes=200)
    if args.model == "wideresnet50":
        net = wide_resnet50_2(num_classes=200)
    if args.model == "wideresnet101":
        net = wide_resnet101_2(num_classes=200)
    if args.model == "resnext50_32x4d":
        net = resnet18(num_classes=200)
    if args.model == "resnext101_32x8d":
        net = resnext101_32x8d(num_classes=200)

# print(net)
net.to(device)

if torch.cuda.device_count() > 0:
    net=nn.DataParallel(net, device_ids=device_ids)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
init = False

if __name__ == "__main__":
    best_acc = 0
    interval = int(len(trainloader)*args.interval_rate)
    #so the only way is to create a new wandb log
    wandb.init(dir='./wandb',entity=args.wandb_entity, project=args.wandb_project, name='self_'+args.model+'_interval'+str(args.interval_rate) + 
                '_batchsize' + str(args.batchsize) + '_' + str(args.dataset), config=args)
    wandb_url = wandb.run.get_url()
    print(f"Wandb URL: {wandb_url}")

    buffer={'inputs':[], 'labels':[]}
    for epoch in range(args.epoch):
        correct = [0 for _ in range(5)]
        predicted = [0 for _ in range(5)]
        # adjust learning rate
        # if epoch in [args.epoch // 3, args.epoch * 2 // 3, args.epoch - 10]:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] /= 10
        net.train()
        sum_loss, total = 0.0, 0.0
        for i, data in enumerate(trainloader, 0):

            n_iter = (epoch - 1) * len(trainloader) + i + 1

            length = len(trainloader)
            inputs, labels = data
            buffer['inputs'].append(inputs)
            buffer['labels'].append(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, outputs_feature = net(inputs)
            ensemble = sum(outputs[:-1])/len(outputs)
            ensemble.detach_()
            
            if interval == 0:
                kd = True
            elif n_iter % interval == 0:
                kd = (n_iter % interval == 0)
            else:
                kd = False

            if init is False:
                #   init the adaptation layers.
                #   we add feature adaptation layers here to soften the influence from feature distillation loss
                #   the feature distillation in our conference version :  | f1-f2 | ^ 2
                #   the feature distillation in the final version : |Fully Connected Layer(f1) - f2 | ^ 2
                layer_list = []
                teacher_feature_size = outputs_feature[0].size(1)
                for index in range(1, len(outputs_feature)):
                    student_feature_size = outputs_feature[index].size(1)
                    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
                net.adaptation_layers = nn.ModuleList(layer_list)
                net.adaptation_layers.cuda()
                optimizer = optim.SGD(net.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
                #   define the optimizer here again so it will optimize the net.adaptation_layers
                init = True       

            #   compute loss
            loss = torch.FloatTensor([0.]).to(device)

            # teacher_output = outputs[0].detach()
            # teacher_feature = outputs_feature[0].detach()

            if kd is True:
                for t_inputs, t_labels in zip(buffer['inputs'], buffer['labels']):
                    t_inputs, t_labels = t_inputs.to(device), t_labels.to(device)
                    loss = torch.FloatTensor([0.]).to(device)
                    if interval == 0:
                        loss += criterion(outputs[0], labels)
                        teacher_output = outputs[0].detach()
                        teacher_feature = outputs_feature[0].detach()
                    else:
                        # with torch.no_grad():
                        outputs, outputs_feature = net(t_inputs)
                        loss += criterion(outputs[0], t_labels)
                        teacher_output = outputs[0].detach()
                        teacher_feature = outputs_feature[0].detach()

                    #   for shallow classifiers
                    for index in range(1, len(outputs)):
                        #   logits distillation
                        loss += CrossEntropy(outputs[index], teacher_output) * args.loss_coefficient
                        loss += criterion(outputs[index], t_labels) * (1 - args.loss_coefficient)
                        #   feature distillation
                        if index != 1:
                            loss += torch.dist(net.adaptation_layers[index-1](outputs_feature[index]), teacher_feature) * \
                                    args.feature_loss_coefficient
                            
                    sum_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += float(labels.size(0))
                    outputs.append(ensemble)

                buffer={'inputs':[], 'labels':[]}

            elif kd is False:
                #   for deepest classifier
                loss += criterion(outputs[0], labels)
                sum_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(labels.size(0))
                outputs.append(ensemble)

            #   the feature distillation loss will not be applied to the shallowest classifier
            # elif space is False:
            #     for index in range(1, len(outputs)):
            #         loss += criterion(outputs[index], labels) * (1 - args.loss_coefficient)

            # sum_loss += loss.item()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # total += float(labels.size(0))
            # outputs.append(ensemble)

            for classifier_index in range(len(outputs)):
                _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: 4/4: %.2f%% 3/4: %.2f%% 2/4: %.2f%%  1/4: %.2f%%'
                  ' Ensemble: %.2f%%' % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                                          100 * correct[0] / total, 100 * correct[1] / total,
                                          100 * correct[2] / total, 100 * correct[3] / total,
                                          100 * correct[4] / total))
            
            wandb.log({'Train/loss': sum_loss / (i + 1), 'Train/acc_4': 100 * correct[0] / total, 
                       'Train/acc_3': 100 * correct[1] / total, 'Train/acc_2': 100 * correct[2] / total, 'Train/acc_1': 100 * correct[3] / total, 'Train/acc_ensemble': 100 * correct[4] / total})

        print("Waiting Test!")
        with torch.no_grad():
            correct = [0 for _ in range(5)]
            predicted = [0 for _ in range(5)]
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs, outputs_feature = net(images)
                ensemble = sum(outputs) / len(outputs)
                outputs.append(ensemble)
                for classifier_index in range(len(outputs)):
                    _, predicted[classifier_index] = torch.max(outputs[classifier_index].data, 1)
                    correct[classifier_index] += float(predicted[classifier_index].eq(labels.data).cpu().sum())
                total += float(labels.size(0))

            print('Test Set AccuracyAcc: 4/4: %.4f%% 3/4: %.4f%% 2/4: %.4f%%  1/4: %.4f%%'
                  ' Ensemble: %.4f%%' % (100 * correct[0] / total, 100 * correct[1] / total,
                                         100 * correct[2] / total, 100 * correct[3] / total,
                                         100 * correct[4] / total))
            wandb.log({'Test/acc_4': 100 * correct[0] / total, 'Test/acc_3': 100 * correct[1] / total, 
                       'Test/acc_2': 100 * correct[2] / total, 'Test/acc_1': 100 * correct[3] / total, 'Test/acc_ensemble': 100 * correct[4] / total})
            
            if correct[4] / total > best_acc:
                best_acc = correct[4]/total
                print("Best Accuracy Updated: ", best_acc * 100)
                # torch.save(net.state_dict(), "./checkpoints/"+str(args.model)+".pth")
            wandb.log({'Best_acc': best_acc})

    print("Training Finished, TotalEPOCH=%d, Best Accuracy=%.3f" % (args.epoch, best_acc))
    wandb.finish()
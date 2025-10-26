import torch
import torch.nn as nn

###############################################################################
#### student and teacher model ####
###############################################################################

class teacher_net(nn.Module):
    def __init__(self, args, num_classes=100):
        super(teacher_net, self).__init__()
        if args.net_teacher =='resnet18':
            from models.resnet import resnet18
            self.net = resnet18(downsample_factor=args.downsample_factor, num_classes = num_classes)
        elif args.net_teacher =='resnet50':
            from models.resnet import resnet50
            self.net = resnet50(num_classes = num_classes)
        elif args.net_teacher =='resnet101':
            from models.resnet import resnet101
            self.net = resnet101(num_classes = num_classes)
        elif args.net_teacher =='resnet152':
            from models.resnet import resnet152
            self.net = resnet152(num_classes = num_classes)
        
        elif args.net_teacher =='mobilenet':
            from models.mobilenet import mobilenet
            self.net = mobilenet(num_classes = num_classes)
        elif args.net_teacher =='mobilenetv2':
            from models.mobilenetv2 import mobilenetv2
            self.net = mobilenetv2(num_classes = num_classes)

        elif args.net_teacher =='shufflenet':
            from models.shufflenet import shufflenet
            self.net = shufflenet(num_classes = num_classes)
        elif args.net_teacher =='shufflenetv2':
            from models.shufflenetv2 import shufflenetv2
            self.net = shufflenetv2(num_classes = num_classes)

        elif args.net_teacher =='vgg11':
            from models.vgg import vgg11_bn
            self.net = vgg11_bn(num_classes = num_classes)
        elif args.net_teacher =='vgg13':
            from models.vgg import vgg13_bn
            self.net = vgg13_bn(num_classes = num_classes)
        elif args.net_teacher =='vgg16':
            from models.vgg import vgg16_bn
            self.net = vgg16_bn(num_classes = num_classes)
        elif args.net_teacher =='vgg19':
            from models.vgg import vgg19_bn
            self.net = vgg19_bn(num_classes = num_classes)

        elif args.net_teacher =='vit':
            from models.vit import ViT
            if args.dataset == 'cifar100':
                self.net = ViT(image_size = 32, patch_size = 8,
                num_classes=num_classes, dim = 384, depth = 6,
                heads = 12, mlp_dim = 384, dropout = 0.1,
                emb_dropout = 0.1, pool='cls')
            elif args.dataset == 'imagenet':
                self.net = ViT(image_size = 224, patch_size = 16,
                num_classes=num_classes, dim = 3072, depth = 12,
                heads = 12, mlp_dim = 3072, dropout = 0.1,
                emb_dropout = 0.1, pool='cls')
    @property
    def feature_size(self):
        return self.net.feature_size

    def forward(self,x):
        feat, out = self.net(x)
        return feat, out
    

class student_net(nn.Module):
    def __init__(self, args, teacher_feature_size, num_classes=100):
        super(student_net, self).__init__()
        if args.net_student =='resnet18':
            from models.resnet import resnet18
            self.net = resnet18(num_classes=num_classes)
        elif args.net_student =='resnet50':
            from models.resnet import resnet50
            self.net = resnet50(num_classes=num_classes)
        elif args.net_student =='resnet101':
            from models.resnet import resnet101
            self.net = resnet101(num_classes=num_classes)
        elif args.net_student =='resnet152':
            from models.resnet import resnet152
            self.net = resnet152(num_classes=num_classes)

        elif args.net_student =='mobilenet':
            from models.mobilenet import mobilenet
            self.net = mobilenet(num_classes = num_classes)
        elif args.net_student =='mobilenetv2':
            from models.mobilenetv2 import mobilenetv2
            self.net = mobilenetv2(num_classes = num_classes)
        
        elif args.net_teacher =='shufflenet':
            from models.shufflenet import shufflenet
            self.net = shufflenet(num_classes = num_classes)
        elif args.net_teacher =='shufflenetv2':
            from models.shufflenetv2 import shufflenetv2
            self.net = shufflenetv2(num_classes = num_classes)

        
        elif args.net_student =='vgg11':
            from models.vgg import vgg11_bn
            self.net = vgg11_bn(num_classes = num_classes)
        elif args.net_student =='vgg13':
            from models.vgg import vgg13_bn
            self.net = vgg13_bn(num_classes = num_classes)
        elif args.net_student =='vgg16':
            from models.vgg import vgg16_bn
            self.net = vgg16_bn(num_classes = num_classes)
        elif args.net_student =='vgg19':
            from models.vgg import vgg19_bn
            self.net = vgg19_bn(num_classes = num_classes)
        
        elif args.net_student =='vit':
            from models.vit import ViT
            if args.dataset == 'cifar100':
                self.net = ViT(image_size = 32, patch_size = 8,
                num_classes = num_classes, dim = 384, depth = 6,
                heads = 12, mlp_dim = 384, dropout = 0.1,
                emb_dropout = 0.1, pool='cls')
            elif args.dataset == 'imagenet':
                self.net = ViT(image_size = 224, patch_size = 16,
                num_classes=num_classes, dim = 3072, depth = 12,
                heads = 12, mlp_dim = 3072, dropout = 0.1,
                emb_dropout = 0.1, pool='cls')

      
        self.linear1 = nn.Linear(self.net.feature_size, int((self.net.feature_size+teacher_feature_size)/2))
        self.linear2 = nn.Linear(int((self.net.feature_size+teacher_feature_size)/2), teacher_feature_size)
        self.relu = nn.ReLU()

    def forward(self,x):
        feat, out = self.net(x)
        feat = self.linear1(feat)
        feat = self.relu(feat)
        feat = self.linear2(feat)
        return feat, out


###############################################################################
#### combine teacher and student model ####
###############################################################################

class spaced_distillation(nn.Module):
    def __init__(self,teacher,student):
        super(spaced_distillation, self).__init__()
        if teacher == 'resnet18':
            self.teacher = resnet18()
        elif teacher =='resnet50':
            self.teacher = resnet50()
        elif teacher =='resnet101':
            self.teacher = resnet101()
        elif teacher =='resnet152':
            self.teacher = resnet152()
        
        if student == 'resnet18':
            self.student = resnet18()
        elif student =='resnet50':
            self.student = resnet50()
        elif student =='resnet101':
            self.student = resnet101()
        elif student =='resnet152':
            self.student = resnet152()

        self.linear1 = nn.Linear(self.student.feature_size, int((self.student.feature_size+self.teacher.feature_size)/2))
        self.linear2 = nn.Linear(int((self.student.feature_size+self.teacher.feature_size)/2), self.teacher.feature_size)
        self.relu = nn.ReLU()

    def forward(self,x,obj):
        if obj=='teacher':
            feat_teacher, out_teacher =self.teacher(x)
            
        elif obj=='student':
            feat_student, out_student =self.student(x)

            feat_student = self.linear1(feat_student)
            feat_student = self.relu(feat_student)
            feat_student = self.linear2(feat_student)
        
        if obj=='teacher':
            return feat_teacher, out_teacher
        elif obj=='student':
            return feat_student, out_student



            









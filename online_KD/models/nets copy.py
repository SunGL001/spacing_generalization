import torch
import torch.nn as nn
import math

###############################################################################
#### resnet ####
###############################################################################
"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, downsample_factor=1):
        super().__init__()

        self.d = math.sqrt(downsample_factor)
        self.block_name = block.__name__

        self.in_channels = int(64/self.d)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(64/self.d), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(64/self.d)),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, int(64/self.d), num_block[0], 1)
        self.conv3_x = self._make_layer(block, int(128/self.d), num_block[1], 2)
        self.conv4_x = self._make_layer(block, int(256/self.d), num_block[2], 2)
        self.conv5_x = self._make_layer(block, int(512/self.d), num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512/self.d) * block.expansion, num_classes)

    @property
    def feature_size(self):
        if self.block_name == 'BasicBlock':
            return int(512/self.d)
        elif self.block_name  == 'BottleNeck':
            return int(512/self.d)*4

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output_feature = output.view(output.size(0), -1)
        # self.feature = output_feature
        output = self.fc(output_feature)

        return output_feature, output


def resnet18(downsample_factor=1, num_classes=100):
    """ return a half-size ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], downsample_factor=downsample_factor, num_classes=num_classes)


def resnet50(downsample_factor=1, num_classes=100):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], downsample_factor=downsample_factor, num_classes=num_classes)


def resnet101(downsample_factor=1, num_classes=100):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], downsample_factor=downsample_factor, num_classes=num_classes)

def resnet152(downsample_factor=1, num_classes=100):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], downsample_factor=downsample_factor, num_classes=num_classes)

###############################################################################
#### vgg ####
###############################################################################
"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
    @property
    def feature_size(self):
        return 512

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size()[0], -1)
        out = self.classifier(feat)

        return feat, out

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(num_class):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_class=num_class)

def vgg13_bn(num_class):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_class=num_class)

def vgg16_bn(num_class):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_class=num_class)

def vgg19_bn(num_class):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_class=num_class)


###############################################################################
#### Vit ####
###############################################################################
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)
        self.dim = dim

    @property
    def feature_size(self):
        return int(self.dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return x, self.mlp_head(x)


###############################################################################
#### student and teacher model ####
###############################################################################

class teacher_net(nn.Module):
    def __init__(self, args, num_classes=100):
        super(teacher_net, self).__init__()
        if args.net_teacher =='resnet18':
            from models.resnet import resnet18
            self.net = resnet18(num_classes = num_classes)
        elif args.net_teacher =='resnet50':
            self.net = resnet50(num_classes = num_classes)
        elif args.net_teacher =='resnet101':
            self.net = resnet101(num_classes = num_classes)
        elif args.net_teacher =='resnet152':
            self.net = resnet152(num_classes = num_classes)

        elif args.net_teacher =='vgg11':
            self.net = vgg11_bn(num_classes = num_classes)
        elif args.net_teacher =='vgg13':
            self.net = vgg13_bn(num_classes = num_classes)
        elif args.net_teacher =='vgg16':
            self.net = vgg16_bn(num_classes = num_classes)
        elif args.net_teacher =='vgg19':
            self.net = vgg19_bn(num_classes = num_classes)

        elif args.net_teacher =='vit':
            if args.dataset == 'cifar100':
                self.net = ViT(image_size = 32, patch_size = 8,
                num_classes=num_classes, dim = 384, depth = 6,
                heads = 12, mlp_dim = 384, dropout = 0.1,
                emb_dropout = 0.1, pool='cls')
            elif args.dataset == 'Imagenet':
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
            self.net = resnet18(num_classes=num_classes)
        elif args.net_student =='resnet50':
            self.net = resnet50(num_classes=num_classes)
        elif args.net_student =='resnet101':
            self.net = resnet101(num_classes=num_classes)
        elif args.net_student =='resnet152':
            self.net = resnet152(num_classes=num_classes)
        
        elif args.net_student =='vgg11':
            self.net = vgg11_bn(num_classes = num_classes)
        elif args.net_student =='vgg13':
            self.net = vgg13_bn(num_classes = num_classes)
        elif args.net_student =='vgg16':
            self.net = vgg16_bn(num_classes = num_classes)
        elif args.net_student =='vgg19':
            self.net = vgg19_bn(num_classes = num_classes)
        
        elif args.net_student =='vit':
            if args.dataset == 'cifar100':
                self.net = ViT(image_size = 32, patch_size = 8,
                num_classes = num_classes, dim = 384, depth = 6,
                heads = 12, mlp_dim = 384, dropout = 0.1,
                emb_dropout = 0.1, pool='cls')
            elif args.dataset == 'Imagenet':
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



            









import torch
from torch import nn
import torchvision
import torchvision.models as models
import math
import numpy as np
# from torch.cuda.amp import autocast # 半精

class ResidualBlock(nn.Module):
    '''
    残差块
    '''
    # k3n64s1
    def __init__(self, input_channel=64, output_channel=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.PReLU(),

            nn.Conv2d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm2d(output_channel)
        )

    # @autocast()  # 半精
    def forward(self, x0):
        x1 = self.layer(x0)
        return x0 + x1

class Generator(nn.Module):
    '''
    生成器
    '''
    def __init__(self, config):
        super().__init__()

        # 放大倍数
        scale = config.scaling_factor

        # k9n64s1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, stride=1, padding=4),
            nn.PReLU()
        )

        # k3n64s1
        conv2_blocks = []        
        for i in range(config.G.n_blocks ):
            conv2_blocks.append(ResidualBlock())
        self.conv2 = nn.Sequential(*conv2_blocks)


        # k3n64s1
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # k3n256s1
        conv4_blocks = []    
        # 一次放大2倍，总共放大 scale//2 次    
        for i in range(int(scale//2)):
            conv4_blocks.append(nn.Conv2d(64, 256, 3, stride=1, padding=1))
            # 亚像素卷积
            conv4_blocks.append(nn.PixelShuffle(2))
            conv4_blocks.append(nn.PReLU())
        self.conv4 = nn.Sequential(*conv4_blocks)  
     
        # k9n3s1
        self.conv5 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    # @autocast()  # 半精
    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x1)
        x = self.conv3(x)
        # conv3的输出+conv1的输出
        x = self.conv4(x + x1)
        x = self.conv5(x)
        return x

class DownBlock(nn.Module):
    '''
    下降块
    '''
    def __init__(self, input_channel, output_channel,  stride, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )

    # @autocast()  # 半精
    def forward(self, x):
        x = self.layer(x)
        return x


class Discriminator(nn.Module):
    '''
    判别器
    '''
    def __init__(self, config):
        super().__init__()
        fc_size = config.D.fc_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, config.D.n_channels, config.D.kernel_size, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        # k3n64s2 -> k3n128s1 -> k3n128s2 -> k3n256s1 -> k3n256s2-> k3n512s1 -> k3n512s2
        conv2_blocks = []   
        conv2_blocks.append(DownBlock(config.D.n_channels, 64, stride=2, padding=1),)            
        # 约定判别器blocks为3份，前1/3层的Channel为123，中间1/3层的Channel为256，后1/3层的Channel为512
        for i in range(config.D.n_blocks ):
            input_channel = 0
            output_channel = 0
            if i == 0 :
                input_channel = config.D.n_channels
                output_channel = 64
            elif i == 1 :
                input_channel = 64
                output_channel = 128
            elif i<= np.floor(config.D.n_blocks*0.333):
                input_channel = 128
                output_channel = 128
            elif i== np.ceil(config.D.n_blocks*0.333):
                input_channel = 128
                output_channel = 256
            elif i<= np.floor(config.D.n_blocks*0.666):
                input_channel = 256
                output_channel = 256
            elif i== np.ceil(config.D.n_blocks*0.666):
                input_channel = 256
                output_channel = 512
            else:
                input_channel = 512
                output_channel = 512
            stride = 2 if input_channel==output_channel else 1
            conv2_blocks.append(DownBlock(input_channel, output_channel, stride=stride, padding=1))
        self.conv2 = nn.Sequential(*conv2_blocks)

        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, fc_size, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(fc_size, 1, 1),
            nn.Sigmoid()
        )

    # @autocast()  # 半精
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class TruncatedVGG19(nn.Module):
    '''
    获取 VGG19 网络中第i层最大池化层前第 j 层卷积(激活后)的特征图。
    '''
    def __init__(self, i, j):
        '''
        i: VGG19 网络中第i层最大池化层
        j:第i层最大池化层前的第 j 层卷积(激活后)之后
        '''
        super(TruncatedVGG19, self).__init__()
        vgg = models.vgg19(True)
        for pa in vgg.parameters():
            pa.requires_grad = False

        i,j=4,5
        if i == 1:
            i=3
        elif i == 2:
            i=8
        elif i == 3:
            i=17
        else:
            i = 25
        print (i + j*2)
        self.vgg = vgg.features[:i + j*2 + 1]

    # @autocast()  # 半精
    def forward(self, x):
        out = self.vgg(x)
        return out

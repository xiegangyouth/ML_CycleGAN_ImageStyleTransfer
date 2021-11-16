"""
CycleGAN模型1:以ResNet为生成器骨干网络
"""
import torch.nn as nn
import torch.nn.functional as F

# 定义残差块
class resBlock(nn.Module):
    def __init__(self, in_channel):
        super(resBlock, self).__init__()

        convBlock = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
        ]

        self.convBlock = nn.Sequential(*convBlock)

    def forward(self, x):
        return x + self.convBlock(x)

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # step1:首个卷积层(7x7卷积)
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # step2:下采样
        in_channel = 64
        out_channel = in_channel * 2
        for i in range(2):
            layers += [
                nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            # 更新输入输出通道数
            in_channel = out_channel
            out_channel = in_channel * 2

        # 构建9个残差块
        for i in range(9):
            layers += [resBlock(in_channel)]


        # step3:上采样
        out_channel = in_channel //2
        for _ in range(2):
            layers += [nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_channel),
                    nn.ReLU(inplace=True)
                    ]
            in_channel = out_channel
            out_channel = in_channel // 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channel, 3, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        layers = [nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)




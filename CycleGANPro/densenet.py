import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils

# 定义密集块的单元层
class Bottleneck(nn.Module):
    def __init__(self, inChannels, k):
        super(Bottleneck, self).__init__()
        outChannels = 4*k
        self.instance_norm2d1 = nn.InstanceNorm2d(inChannels)# 标准化
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, bias=False)
        self.instance_norm2d2 = nn.InstanceNorm2d(outChannels)# 标准化
        self.conv2 = nn.Conv2d(outChannels, k, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.instance_norm2d1(x))) #3x3卷积前增加1x1卷积:减少参数量
        out = self.conv2(F.relu(self.instance_norm2d2(out))) # 每一个Bottleneck块单独产生k个特征图
        out = torch.cat((x, out), 1) #产出通道数与输入通道数合并
        return out

# 定义连接层(衰减层)
class Transition(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Transition, self).__init__()
        self.instance_norm2d1 = nn.InstanceNorm2d(inchannels)
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.instance_norm2d1(x)))
        return out

# 定义DenseNet-BC结构
class DenseNet(nn.Module):
    def __init__(self, k, depth, reduction):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3 // 2 # 计算密集块个数

        nChannels = 2*k
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=7, padding=3, bias=False)
        # 密集块1
        self.dense1 = self.makeDenseBlock(nChannels, k, nDenseBlocks)
        # 转换层1
        nChannels += nDenseBlocks*k # 每一个密集块总共输出的通道数
        nOutChannels = int(math.floor(nChannels*reduction)) # 衰减系数=0.5
        self.trans1 = Transition(nChannels, nOutChannels)
        # 密集块2
        nChannels = nOutChannels
        self.dense2 = self.makeDenseBlock(nChannels, k, nDenseBlocks)
        # 转换层2
        nChannels += nDenseBlocks*k
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        # 转换块3
        nChannels = nOutChannels
        self.dense3 = self.makeDenseBlock(nChannels, k, nDenseBlocks)
        # 转换层3
        nChannels += nDenseBlocks*k
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans3 = Transition(nChannels, nOutChannels)

        self.instance_norm2d = nn.InstanceNorm2d(nChannels)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    # 构造密集块
    def makeDenseBlock(self, nChannels, k, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(Bottleneck(nChannels, k))
            nChannels += k
        return nn.Sequential(*layers)

    def forward(self, x):
        # step1:首个卷积层
        out = self.conv1(x)
        # step2:密集块1+衰减层1
        out = self.trans1(self.dense1(out))
        # step3:密集块2+衰减层2
        out = self.trans2(self.dense2(out))
        # step4:密集块3+衰减层3
        out = self.trans3(self.dense3(out))

        # 将通道数恢复为3
        ins = out.size()[1]
        fun1 = nn.Conv2d(ins, 30, 1)
        fun2 = nn.Conv2d(30,15, 1)
        fun3 = nn.Conv2d(15, 3, 1)
        out = F.relu(self.instance_norm2d(fun1(out)))
        out = F.relu(self.instance_norm2d(fun2(out)))
        out = F.relu(self.instance_norm2d(fun3(out)))

        return out

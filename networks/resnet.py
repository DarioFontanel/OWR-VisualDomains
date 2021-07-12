import math
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ncm import NCM_classifier


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, relu=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = DownsampleA(in_planes, planes * self.expansion, stride)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        if self.relu:
            out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, relu=True, deep_nno=False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, last=not relu)
        self.linear = NCM_classifier(64, num_classes, deep_nno)
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(1. / 64.))
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, last=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        if last:
            for stride in strides[:-1]:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            layers.append(block(self.in_planes, planes, strides[-1], relu=False))
            self.in_planes = planes * block.expansion
        else:
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))

                self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out_before_pooling = self.layer3(out)
        out = out_before_pooling.mean(-1).mean(-1)
        out = out.view(out.size(0), -1)
        return out, out_before_pooling

    def update_means(self, x, y):
        self.linear.update_means(x, y)

    def predict(self, x):
        out = self.linear(x)
        return out

    def add_classes(self, new_classes):
        if new_classes == 0:
            return
        self.linear.add_classes(new_classes)
        self.num_classes += new_classes


def ResNet18(classes, pretrained, relu=True, deep_nno=False):
    net = ResNet(BasicBlock, [3, 3, 3], num_classes=classes, relu=relu, deep_nno=deep_nno)
    if pretrained is not None:
        net.load_state_dict(pretrained['state_dict'])
    return net

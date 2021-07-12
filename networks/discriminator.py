import math
from torch.nn import init
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, batch_size, n_feat, n_classes):
        super(Discriminator, self).__init__()
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.linear = nn.Linear(batch_size*16*16, self.n_feat)
        self.bn = nn.BatchNorm1d(self.n_feat)
        self.leaky= nn.LeakyReLU(0.2, inplace=True)
        self.classify = nn.Linear(self.n_feat, self.n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(1. / 64.))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.leaky(out)
        out = self.classify(out)
        return out

import torch.nn as nn
from torch.nn.functional import relu


# Margin loss definition
class MarginLoss(nn.Module):

    def __init__(self):
        super(MarginLoss, self).__init__()

    def forward(self, x, y, tau):
        return relu((-1)**y * (tau - x)).sum(dim=1).mean()

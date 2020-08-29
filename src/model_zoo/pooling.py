import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class AdaptiveGlobalPool2d(nn.modules.pooling._AdaptiveMaxPoolNd):
    def __init__(self, output_size, return_indices=False):
        super().__init__(output_size, return_indices)
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, x):
        max_ = F.adaptive_max_pool2d(x, self.output_size, self.return_indices)
        avg_ = F.adaptive_max_pool2d(x, self.output_size, self.return_indices)
        return torch.cat([max_, avg_], 1)
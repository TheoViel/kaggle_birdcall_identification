import torch
import torch.nn as nn
import torch.nn.functional as F
import resnest.torch as resnest_torch

from efficientnet_pytorch import EfficientNet

from params import DEVICE

# from model_zoo.common import SETTINGS, get_encoder
from model_zoo import msd_resnest
from model_zoo import msd_resnext


class AdaptiveGlobalPool2d(nn.modules.pooling._AdaptiveMaxPoolNd):
    def __init__(self, output_size, return_indices=False):
        super().__init__(output_size, return_indices)
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, x):
        max_ = F.adaptive_max_pool2d(x, self.output_size, self.return_indices)
        avg_ = F.adaptive_max_pool2d(x, self.output_size, self.return_indices)
        return torch.cat([max_, avg_], 1)


def get_model(name, use_msd=False, num_classes=1):
    if use_msd:
        if "resnest" in name:
            model = getattr(msd_resnest, name)(pretrained=True)
        elif "resnext" in name:
            model = getattr(msd_resnext, name)()
        else:
            raise NotImplementedError
    else:
        if "resnest" in name:
            model = getattr(resnest_torch, name)(pretrained=True)
        elif "resnext" in name:
            model = torch.hub.load("facebookresearch/WSL-Images", name)
        elif "resnet" in name:
            model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
        elif "efficientnet" in name:
            model = EfficientNet.from_pretrained(name)
        else:
            raise NotImplementedError

    if "efficientnet" not in name:
        nb_ft = model.fc.in_features
        del model.fc
        model.fc = nn.Linear(nb_ft, num_classes)
    else:
        nb_ft = model._fc.in_features
        del model._fc
        model._fc = nn.Linear(nb_ft, num_classes)

    # model.avgpool = AdaptiveGlobalPool2d((1, 1))
    # model.avgpool = nn.AdaptiveMaxPool2d((1, 1))

    return model

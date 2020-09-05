import torch
import torchvision
import torch.nn as nn
import pretrainedmodels
import resnest.torch as resnest_torch

from efficientnet_pytorch import EfficientNet

from params import DEVICE
from model_zoo import msd_resnest
from model_zoo import msd_resnext
from model_zoo.pooling import GeM, AdaptiveGlobalPool2d


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
        elif "wsl" in name:
            model = torch.hub.load("facebookresearch/WSL-Images", name, pretrained=True)
        elif "se" in name:
            model = pretrainedmodels.__dict__[name](num_classes=1000, pretrained='imagenet')
        elif "resnext" in name or "resnet" in name:
            model = torch.hub.load('pytorch/vision:v0.6.0', name, pretrained=True)
        elif "efficientnet" in name:
            model = EfficientNet.from_pretrained(name)
        elif "inception" in name:
            model = torchvision.models.inception_v3(pretrained=True)
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

    # model.avgpool = GeM()
    # model.avgpool = nn.AdaptiveMaxPool2d((1, 1))

    return model

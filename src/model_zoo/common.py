import torch
import torch.nn as nn
import torch.nn.functional as F

from model_zoo.resnet import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
from pretrainedmodels.models.torchvision_models import pretrained_settings

SETTINGS = {
    "resnet18": {
        "name": "resnet18",
        "encoder": ResNet,
        "pretrained_settings": pretrained_settings["resnet18"]["imagenet"],
        "out_shapes": (512, 256, 128, 64, 64),
        "params": {"block": BasicBlock, "layers": [2, 2, 2, 2],},
    },
    "resnet34": {
        "name": "resnet34",
        "encoder": ResNet,
        "pretrained_settings": pretrained_settings["resnet34"]["imagenet"],
        "out_shapes": (512, 256, 128, 64, 64),
        "params": {"block": BasicBlock, "layers": [3, 4, 6, 3],},
    },
    "resnet50": {
        "name": "resnet50",
        "encoder": ResNet,
        "pretrained_settings": pretrained_settings["resnet50"]["imagenet"],
        "out_shapes": (2048, 1024, 512, 256, 64),
        "params": {"block": Bottleneck, "layers": [3, 4, 6, 3],},
    },
    "resnet101": {
        "name": "resnet101",
        "encoder": ResNet,
        "pretrained_settings": pretrained_settings["resnet101"]["imagenet"],
        "out_shapes": (2048, 1024, 512, 256, 64),
        "params": {"block": Bottleneck,"layers": [3, 4, 23, 3],},
    },
} 


def get_encoder(settings):
    """
    Builds a CNN architecture settings["encoder"] using settings["params"],
    and loads the pretrained weight from settings["pretrained_settings"]["url"]
    Implemented only for some ResNets here

    Arguments:
        settings {dict} -- Settings dictionary associated to a model
    
    Returns:
        Pretrained model
    """
    Encoder = settings["encoder"]
    encoder = Encoder(**settings["params"])
    encoder.out_shapes = settings["out_shapes"]

    if settings["pretrained_settings"] is not None:
        encoder.load_state_dict(
            torch.utils.model_zoo.load_url(settings["pretrained_settings"]["url"])
        )
    return encoder






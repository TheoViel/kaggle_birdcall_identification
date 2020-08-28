import torchvision

from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import Bottleneck, BasicBlock


model_urls = {
    "resnext101_32x8d": "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth",
    "resnext101_32x16d": "https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth",
    "resnext101_32x32d": "https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth",
    "resnext101_32x48d": "https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth",
}


class ResNet(torchvision.models.resnet.ResNet):
    """
    Slightly modified torchvision ResNet.
    The last fully connected layer was removed for a more convenient use
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4

    def load_state_dict(self, state_dict, **kwargs):
        try:
            state_dict.pop("fc.bias")
            state_dict.pop("fc.weight")
        except KeyError:
            pass
        super().load_state_dict(state_dict, **kwargs)


def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    """
    [Taken from https://github.com/facebookresearch/WSL-Images]
    """
    model = ResNet(block, layers, **kwargs)
    state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    model.load_state_dict(state_dict)
    return model


def resnext101_32x8d_wsl(progress=True, **kwargs):
    """
    [Taken from https://github.com/facebookresearch/WSL-Images]
    Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnext(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], True, progress, **kwargs
    )


def resnext101_32x16d_wsl(progress=True, **kwargs):
    """
    [Taken from https://github.com/facebookresearch/WSL-Images]
    Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 16
    return _resnext(
        "resnext101_32x16d", Bottleneck, [3, 4, 23, 3], True, progress, **kwargs
    )


def resnext101_32x32d_wsl(progress=True, **kwargs):
    """
    [Taken from https://github.com/facebookresearch/WSL-Images]
    Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 32
    return _resnext(
        "resnext101_32x32d", Bottleneck, [3, 4, 23, 3], True, progress, **kwargs
    )


def resnext101_32x48d_wsl(progress=True, **kwargs):
    """
    [Taken from https://github.com/facebookresearch/WSL-Images]
    Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 48
    return _resnext(
        "resnext101_32x48d", Bottleneck, [3, 4, 23, 3], True, progress, **kwargs
    )

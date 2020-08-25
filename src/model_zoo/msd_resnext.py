import torch
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck


wsl_model_urls = {
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}


class MSDResNet(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.training:
            logits = torch.mean(
                torch.stack([self.fc(F.dropout(x, p=0.5, training=True)) for _ in range(4)], dim=0,), dim=0
            )
        else:
            logits = self.fc(x)

        return logits


def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    """
    [Taken from https://github.com/facebookresearch/WSL-Images]
    """
    model = MSDResNet(block, layers, **kwargs)
    state_dict = load_state_dict_from_url(wsl_model_urls[arch], progress=progress)
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
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


def resnext101_32x16d_wsl(progress=True, **kwargs):
    """
    [Taken from https://github.com/facebookresearch/WSL-Images]
    Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


def resnext101_32x32d_wsl(progress=True, **kwargs):
    """
    [Taken from https://github.com/facebookresearch/WSL-Images]
    Constructs a ResNeXt-101 32x32 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 32
    return _resnext('resnext101_32x32d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


def resnext101_32x48d_wsl(progress=True, **kwargs):
    """
    [Taken from https://github.com/facebookresearch/WSL-Images]
    Constructs a ResNeXt-101 32x48 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 48
    return _resnext('resnext101_32x48d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)
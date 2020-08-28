import torch
from resnest.torch.resnet import *
from resnest.torch.ablation import resnest_model_urls


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
                torch.stack([self.fc(self.drop(x)) for _ in range(4)], dim=0,), dim=0
            )
        else:
            logits = self.fc(x)

        return logits


def resnest50_fast_1s1x64d(pretrained=False, root="~/.encoding/models", **kwargs):
    model = MSDResNet(
        Bottleneck,
        [3, 4, 6, 3],
        radix=1,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        final_drop=0.5,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                resnest_model_urls["resnest50_fast_1s1x64d"],
                progress=True,
                check_hash=True,
            )
        )
    return model

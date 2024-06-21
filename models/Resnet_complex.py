import torch
from torch import nn
from einops import rearrange
from utils_complex.complex_functions_bio import stable_angle
import utils_complex.complex_functions_fccn as cf
from typing import Type, Any, Callable, Union, List, Optional
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = cf.ComplexConv2d(inplanes, width, kernel_size=1, stride=1).to(device)
        self.bn1 = norm_layer(width).to(device)
        self.conv2 = cf.ComplexConv2d(width, width, kernel_size=3, stride=self.stride, padding=1).to(device)
        self.bn2 = norm_layer(width).to(device)
        self.conv3 = cf.ComplexConv2d(width, planes * self.expansion, kernel_size=1, stride=1).to(device)
        self.bn3 = norm_layer(planes * self.expansion).to(device)
        self.prelu = nn.ReLU().to(device)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.prelu(self.bn1(out.abs())) * torch.exp(stable_angle(out) * 1j).to(device)

        out = self.conv2(out)
        out = self.prelu(self.bn2(out.abs())) * torch.exp(stable_angle(out) * 1j).to(device)

        out = self.conv3(out)
        out = self.bn3(out.abs()) * torch.exp(stable_angle(out) * 1j).to(device)

        if self.downsample is not None:
            
            identity = self.downsample(x)
        out += identity
        out = self.prelu(out.abs()) * torch.exp(stable_angle(out) * 1j).to(device)

        return out

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = cf.ComplexConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1).to(device)
        self.bn1 = norm_layer(planes).to(device)
        self.prelu = nn.ReLU().to(device)
        self.conv2 = cf.ComplexConv2d(planes, planes, kernel_size=3, stride=1, padding=1).to(device)
        self.bn2 = norm_layer(planes).to(device)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.prelu(self.bn1(out.abs())) * torch.exp(stable_angle(out) * 1j).to(device)
        out = self.conv2(out)
        out = self.bn2(out.abs()) * torch.exp(stable_angle(out) * 1j).to(device)
        if self.downsample is not None:
            identity = self.downsample(x)
            print("downsampling", identity.shape)
        out += identity
        out = self.prelu(out.abs()) * torch.exp(stable_angle(out) * 1j).to(device)
        return out

class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = cf.RealToComplexConvolution2D(3, self.inplanes, kernel_size=7, stride=2, padding=3).to(device)
        self.bn1 = norm_layer(self.inplanes).to(device)
        self.prelu = nn.ReLU().to(device)
        self.maxpool = cf.ComplexMaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
        self.layer1 = self._make_layer(block, 64, layers[0]).to(device)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0]).to(device)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], kernel_size=2, padding=1).to(device)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], kernel_size=2, padding=1).to(device)
        self.avgpool = cf.AdaptiveAvgPoolToConv(kernel_size=1).to(device)
        self.fc = cf.ComplexConv2d(512 * block.expansion, num_classes, kernel_size=1).to(device)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, kernel_size=3, padding=1) :
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = cf.Sequential(
                cf.ComplexConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride).to(device),
                norm_layer(planes * block.expansion).to(device)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer).to(device))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer).to(device))

        return cf.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.prelu(self.bn1(x.abs())) * torch.exp(stable_angle(x) * 1j).to(device)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        pass
        # state_dict = load_state_dict_from_url(model_urls[arch],
                                              # progress=progress)
        # model.load_state_dict(state_dict)
    return model



def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)



def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

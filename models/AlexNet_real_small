import torch
from torch import nn
from einops import rearrange

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes= num_classes
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
                #nn.BatchNorm2d(num_features=192),
                nn.ReLU(),)
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                #nn.BatchNorm2d(num_features=500),
                nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(num_features=1024),
                nn.ReLU(),)
        self.conv4 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(num_features=1500),
                nn.ReLU(),)
        self.conv5 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(num_features=128),
                nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 4096, kernel_size= 3),
                nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(4096, 4096, kernel_size= 1),
                nn.ReLU())
        self.conv8 = nn.Conv2d(4096, self.num_classes, kernel_size= 1)

    def forward(self, x: torch.Tensor) :
        c1 = self.conv1(x)
        mx1 = self.maxpool(c1)
        c2 = self.conv2(mx1)
        mx2 = self.maxpool(c2)
        c3 = self.conv3(mx2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        mx5 = self.maxpool(c5)
        avg = self.avgpool(mx5)
        c6 = self.conv6(avg)
        c7 = self.conv7(c6)
        c8 = self.conv8(c7)
        r = rearrange(c8, 'b c h w -> b (c h w) ')
        return c1, mx1, c2, mx2, c3, c4, c5, mx5, avg, c6, c7, c8, r

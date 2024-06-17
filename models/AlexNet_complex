import torch
from torch import nn
import pytorch_lightning as pl
from einops import rearrange
from utils_complex.complex_functions_bio import stable_angle
import utils_complex.complex_functions_fccn as cf

class ComplexWeigth_AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes= num_classes
        
        self.maxpool = cf.ComplexMaxPool2d(kernel_size=3, stride=2)
        #self.avgpool = cf.ComplexAdaptiveAvgPool2d((3, 3))
        self.avgpool = cf.ComplexConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        #self.conv1 = cf.ComplexConv2d(3, 96, kernel_size=5, stride=1, padding=2)
        self.conv1 = cf.RealToComplexConvolution2D(3, 96, kernel_size=5, stride=1, padding=2)
        self.activ1 = nn.Sequential(nn.BatchNorm2d(num_features=96),
                nn.ReLU())
                
        self.conv2 = cf.ComplexConv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.activ2 = nn.Sequential(nn.BatchNorm2d(num_features=256),
                nn.ReLU(),)
                
        self.conv3 = cf.ComplexConv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.activ3 = nn.Sequential(nn.BatchNorm2d(num_features=384),
                nn.ReLU(),)
                
        self.conv4 = cf.ComplexConv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.activ4 = nn.Sequential(nn.BatchNorm2d(num_features=384),
                nn.ReLU(),)
                
        self.conv5 = cf.ComplexConv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.activ5 = nn.Sequential(nn.BatchNorm2d(num_features=256),
                nn.ReLU(),)
        
        self.conv6 = cf.ComplexConv2d(256, 4096, kernel_size= 3)
        self.activ6 = nn.Sequential(nn.BatchNorm2d(num_features=4096),
                nn.ReLU(),)
                
        self.conv7 = cf.ComplexConv2d(4096, 4096, kernel_size= 1)
        self.activ7 = nn.Sequential(nn.BatchNorm2d(num_features=4096),
                nn.ReLU(),)
                
        self.conv8 = cf.ComplexConv2d(4096, self.num_classes, kernel_size= 1)


    def forward(self, x: torch.Tensor) :
        """
        phase = stable_angle(c1)
        c1 = self.activ1(c1.abs()) 
        mx1_magnitude = self.maxpool(c1)
        mx1_phase = cf.maxpool2d_index(c1, mx1_magnitude, phase, 2, 3)
        
        mx1 = mx1_magnitude * torch.exp(mx1_phase * 1j)
        phase = stable_angle(c2)
        c2 = self.activ2(c2.abs()) 
        mx2_magnitude = self.maxpool(c2)
        mx2_phase = cf.maxpool2d_index(c2, mx2_magnitude, phase, 2, 3)
        
        mx2 = mx2_magnitude * torch.exp(mx2_phase * 1j)
        phase = stable_angle(c5)
        c5 = self.activ5(c5.abs()) 
        mx5_magnitude = self.maxpool(c5)
        mx5_phase = cf.maxpool2d_index(c5, mx5_magnitude, phase, 2, 3)
        
        mx5 = mx5_magnitude * torch.exp(mx5_phase * 1j)
        """
        c1 = self.conv1(x)
        c1 = self.activ1(c1.abs()) * torch.exp(stable_angle(c1) * 1j)
        mx1 = self.maxpool(c1)
        c2 = self.conv2(mx1)
        c2 = self.activ2(c2.abs()) * torch.exp(stable_angle(c2) * 1j)
        mx2 = self.maxpool(c2)
        c3 = self.conv3(mx2)
        c3 = self.activ3(c3.abs()) * torch.exp(stable_angle(c3) * 1j)
        c4 = self.conv4(c3)
        c4 = self.activ4(c4.abs()) * torch.exp(stable_angle(c4) * 1j)
        c5 = self.conv5(c4)
        c5 = self.activ5(c5.abs()) * torch.exp(stable_angle(c5) * 1j)
        mx5 = self.maxpool(c5)
        avg = self.avgpool(mx5)
        c6 = self.conv6(avg)
        c6 = self.activ6(c6.abs()) * torch.exp(stable_angle(c6) * 1j)
        c7 = self.conv7(c6)
        c7 = self.activ7(c7.abs()) * torch.exp(stable_angle(c7) * 1j)
        c8 = self.conv8(c7)
        r = rearrange(c8, 'b c h w -> b (c h w) ')
        return c1, mx1, c2, mx2, c3, c4, c5, mx5, avg, c6, c7, c8, r
        
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes= num_classes
        
        self.maxpool = cf.ComplexMaxPool2d(kernel_size=3, stride=2)
        self.avgpool = cf.ComplexConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.conv1 = cf.RealToComplexConvolution2D(3, 256, kernel_size=5, stride=1, padding=2)
        self.activ1 = nn.Sequential(nn.BatchNorm2d(num_features=256),
                nn.ReLU())
                
        self.conv2 = cf.ComplexConv2d(256, 512, kernel_size=5, stride=1, padding=2)
        self.activ2 = nn.Sequential(nn.BatchNorm2d(num_features=512),
                nn.ReLU(),)
                
        self.conv3 = cf.ComplexConv2d(512, 768, kernel_size=3, stride=1, padding=1)
        self.activ3 = nn.Sequential(nn.BatchNorm2d(num_features=768),
                nn.ReLU(),)
                
        self.conv4 = cf.ComplexConv2d(768, 768, kernel_size=3, stride=1, padding=1)
        self.activ4 = nn.Sequential(nn.BatchNorm2d(num_features=768),
                nn.ReLU(),)
                
        self.conv5 = cf.ComplexConv2d(768, 512, kernel_size=3, stride=1, padding=1)
        self.activ5 = nn.Sequential(nn.BatchNorm2d(num_features=512),
                nn.ReLU(),)
        
        self.conv6 = cf.ComplexConv2d(512, 5120, kernel_size= 3)
        self.activ6 = nn.Sequential(nn.BatchNorm2d(num_features=5120),
                nn.ReLU(),)
                
        self.conv7 = cf.ComplexConv2d(5120, 4096, kernel_size= 1)
        self.activ7 = nn.Sequential(nn.BatchNorm2d(num_features=4096),
                nn.ReLU(),)
                
        self.conv8 = cf.ComplexConv2d(4096, self.num_classes, kernel_size= 1)


    def forward(self, x: torch.Tensor) :
        c1 = self.conv1(x)
        c1 = self.activ1(c1.abs()) * torch.exp(stable_angle(c1) * 1j)
        mx1 = self.maxpool(c1)
        c2 = self.conv2(mx1)
        c2 = self.activ2(c2.abs()) * torch.exp(stable_angle(c2) * 1j)
        mx2 = self.maxpool(c2)
        c3 = self.conv3(mx2)
        c3 = self.activ3(c3.abs()) * torch.exp(stable_angle(c3) * 1j)
        c4 = self.conv4(c3)
        c4 = self.activ4(c4.abs()) * torch.exp(stable_angle(c4) * 1j)
        c5 = self.conv5(c4)
        c5 = self.activ5(c5.abs()) * torch.exp(stable_angle(c5) * 1j)
        mx5 = self.maxpool(c5)
        avg = self.avgpool(mx5)
        c6 = self.conv6(avg)
        c6 = self.activ6(c6.abs()) * torch.exp(stable_angle(c6) * 1j)
        c7 = self.conv7(c6)
        c7 = self.activ7(c7.abs()) * torch.exp(stable_angle(c7) * 1j)
        c8 = self.conv8(c7)
        r = rearrange(c8, 'b c h w -> b (c h w) ')
        return c1, mx1, c2, mx2, c3, c4, c5, mx5, avg, c6, c7, c8, r

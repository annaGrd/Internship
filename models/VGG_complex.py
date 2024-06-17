import torch
from torch import nn
from einops import rearrange
from utils_complex.complex_functions_bio import stable_angle
import utils_complex.complex_functions_fccn as cf

cfg = {'Vgg11':[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'Vgg13':[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'Vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'Vgg19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

class VGG11(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        self.maxpool = cf.ComplexMaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = cf.RealToComplexConvolution2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = cf.ComplexConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = cf.ComplexConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = cf.ComplexConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = cf.ComplexConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = cf.ComplexConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = cf.ComplexConv2d(512, 512, kernel_size=2, stride=1, padding=0)
        
        self.cl1 = cf.ComplexConv2d(512, 4096, kernel_size=1)
        self.cl2 = cf.ComplexConv2d(4096, 4096, kernel_size=1)
        self.cl3 = cf.ComplexConv2d(4096, 10, kernel_size=1)
        
        self.activ64 = nn.Sequential(nn.BatchNorm2d(num_features=64), nn.ReLU())
        self.activ128 = nn.Sequential(nn.BatchNorm2d(num_features=128), nn.ReLU())
        self.activ256 = nn.Sequential(nn.BatchNorm2d(num_features=256), nn.ReLU())
        self.activ512 = nn.Sequential(nn.BatchNorm2d(num_features=512), nn.ReLU())
        self.activ4096 = nn.Sequential(nn.BatchNorm2d(num_features=4096), nn.ReLU())
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.activ64(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv2(self.maxpool(x))
        x = self.activ128(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv3(self.maxpool(x))
        x = self.activ256(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv4(x)
        x = self.activ256(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv5(self.maxpool(x))
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(self.maxpool(x))
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv7(x)  # Output should be (batch_size, 512, 1, 1)
        x = self.cl1(x)
        x = self.activ4096(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.cl2(x)
        x = self.activ4096(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.cl3(x)  # Output should be (batch_size, num_classes, 1, 1)
        x = rearrange(x, 'b c h w -> b (c h w) ')  # Reshape to (batch_size, num_classes)
        return x

class VGG13(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        self.maxpool = cf.ComplexMaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = cf.RealToComplexConvolution2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = cf.ComplexConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = cf.ComplexConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = cf.ComplexConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = cf.ComplexConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = cf.ComplexConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = cf.ComplexConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = cf.ComplexConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = cf.ComplexConv2d(512, 512, kernel_size=2, stride=1, padding=0)
        
        self.cl1 = cf.ComplexConv2d(512, 4096, kernel_size=1)
        self.cl2 = cf.ComplexConv2d(4096, 4096, kernel_size=1)
        self.cl3 = cf.ComplexConv2d(4096, 10, kernel_size=1)
        
        self.activ64 = nn.Sequential(nn.BatchNorm2d(num_features=64), nn.ReLU())
        self.activ128 = nn.Sequential(nn.BatchNorm2d(num_features=128), nn.ReLU())
        self.activ256 = nn.Sequential(nn.BatchNorm2d(num_features=256), nn.ReLU())
        self.activ512 = nn.Sequential(nn.BatchNorm2d(num_features=512), nn.ReLU())
        self.activ4096 = nn.Sequential(nn.BatchNorm2d(num_features=4096), nn.ReLU())
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.activ64(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv1_1(x)
        x = self.activ64(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv2(self.maxpool(x))
        x = self.activ128(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv2_2(x)
        x = self.activ128(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv3(self.maxpool(x))
        x = self.activ256(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv4(x)
        x = self.activ256(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv5(self.maxpool(x))
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(self.maxpool(x))
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv7(x)  # Output should be (batch_size, 512, 1, 1)
        x = self.cl1(x)
        x = self.activ4096(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.cl2(x)
        x = self.activ4096(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.cl3(x)  # Output should be (batch_size, num_classes, 1, 1)
        x = rearrange(x, 'b c h w -> b (c h w) ')  # Reshape to (batch_size, num_classes)
        return x
        
class VGG16(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        self.maxpool = cf.ComplexMaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = cf.RealToComplexConvolution2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = cf.ComplexConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = cf.ComplexConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = cf.ComplexConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = cf.ComplexConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = cf.ComplexConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = cf.ComplexConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = cf.ComplexConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = cf.ComplexConv2d(512, 512, kernel_size=2, stride=1, padding=0)
        
        self.cl1 = cf.ComplexConv2d(512, 4096, kernel_size=1)
        self.cl2 = cf.ComplexConv2d(4096, 4096, kernel_size=1)
        self.cl3 = cf.ComplexConv2d(4096, 10, kernel_size=1)
        
        self.activ64 = nn.Sequential(nn.BatchNorm2d(num_features=64), nn.ReLU())
        self.activ128 = nn.Sequential(nn.BatchNorm2d(num_features=128), nn.ReLU())
        self.activ256 = nn.Sequential(nn.BatchNorm2d(num_features=256), nn.ReLU())
        self.activ512 = nn.Sequential(nn.BatchNorm2d(num_features=512), nn.ReLU())
        self.activ4096 = nn.Sequential(nn.BatchNorm2d(num_features=4096), nn.ReLU())
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.activ64(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv1_1(x)
        x = self.activ64(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv2(self.maxpool(x))
        x = self.activ128(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv2_2(x)
        x = self.activ128(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv3(self.maxpool(x))
        x = self.activ256(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv4(x)
        x = self.activ256(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv4(x)
        x = self.activ256(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv5(self.maxpool(x))
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(self.maxpool(x))
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv7(x)  # Output should be (batch_size, 512, 1, 1)
        x = self.cl1(x)
        x = self.activ4096(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.cl2(x)
        x = self.activ4096(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.cl3(x)  # Output should be (batch_size, num_classes, 1, 1)
        x = rearrange(x, 'b c h w -> b (c h w) ')  # Reshape to (batch_size, num_classes)
        return x     

class VGG19(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        self.maxpool = cf.ComplexMaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = cf.RealToComplexConvolution2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = cf.ComplexConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = cf.ComplexConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = cf.ComplexConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = cf.ComplexConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = cf.ComplexConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = cf.ComplexConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = cf.ComplexConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = cf.ComplexConv2d(512, 512, kernel_size=2, stride=1, padding=0)
        
        self.cl1 = cf.ComplexConv2d(512, 4096, kernel_size=1)
        self.cl2 = cf.ComplexConv2d(4096, 4096, kernel_size=1)
        self.cl3 = cf.ComplexConv2d(4096, 10, kernel_size=1)
        
        self.activ64 = nn.Sequential(nn.BatchNorm2d(num_features=64), nn.ReLU())
        self.activ128 = nn.Sequential(nn.BatchNorm2d(num_features=128), nn.ReLU())
        self.activ256 = nn.Sequential(nn.BatchNorm2d(num_features=256), nn.ReLU())
        self.activ512 = nn.Sequential(nn.BatchNorm2d(num_features=512), nn.ReLU())
        self.activ4096 = nn.Sequential(nn.BatchNorm2d(num_features=4096), nn.ReLU())
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.activ64(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv1_1(x)
        x = self.activ64(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv2(self.maxpool(x))
        x = self.activ128(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv2_2(x)
        x = self.activ128(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv3(self.maxpool(x))
        x = self.activ256(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv4(x)
        x = self.activ256(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv4(x)
        x = self.activ256(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv4(x)
        x = self.activ256(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv5(self.maxpool(x))
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(self.maxpool(x))
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv6(x)
        x = self.activ512(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.conv7(x)  # Output should be (batch_size, 512, 1, 1)
        x = self.cl1(x)
        x = self.activ4096(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.cl2(x)
        x = self.activ4096(x.abs()) * torch.exp(stable_angle(x) * 1j)
        x = self.cl3(x)  # Output should be (batch_size, num_classes, 1, 1)
        x = rearrange(x, 'b c h w -> b (c h w) ')  # Reshape to (batch_size, num_classes)
        return x     

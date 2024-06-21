import math
import numpy as np

import torch
from typing import TypeVar, Union, Tuple, Optional
from torch import nn, Tensor
from torch.nn import Module, Parameter, init
import pytorch_lightning as pl
import torch.nn.functional as F
from utils_complex.complex_functions_bio import stable_angle, retrieve_elements_from_indices, get_complex_number, apply_layer_from_real
from torch.overrides import has_torch_function, handle_torch_function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def is_conv_layer(layer):
    return isinstance(layer, (nn.Conv2d, ComplexConv2d))

def is_bn_or_relu_layer(layer):
    return isinstance(layer, (nn.BatchNorm2d, nn.ReLU))

class Sequential(nn.Module):
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            if is_conv_layer(layer):
                x = layer(x)
            elif is_bn_or_relu_layer(layer):
                x = layer(x.abs()) * torch.exp(stable_angle(x) * 1j)
            else:
                x = layer(x)
        return x


class ComplexConv2d(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride= 1,
        padding = 0,
        dilation= 1,
        groups:int = 1,
        bias: bool = False,
        complex_axis= 1,
        padding_mode: str = 'zeros',
        device= None,
        dtype= None
        ) -> None:
        super().__init__()

        # # check condition that the in_channels are even
        # if (in_channels % 2 != 0) or (out_channels % 2 != 0):
        #     raise ValueError(f"in_channels and out_channels should be even. Got {in_channels} and {out_channels}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel_size = kernel_size
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, stride= stride, padding= padding, dilation= dilation, groups= groups, bias= bias)
        #self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, stride= stride, padding= padding, dilation= dilation, groups= groups, bias= bias)

    def forward(self, x):
        ''' define how the forward prop will take place '''
        # check if the input is of dtype complex
        # for this we can use is_complex() function which will return true if the input is complex dtype
        if not x.is_complex():
            raise ValueError(f"Input should be a complex tensor. Got {x.dtype}")
        return apply_complex(self.conv_real, self.conv_real, x) 
        #return apply_complex(self.conv_real, self.conv_imag, x) 

class RealToComplexConvolution2D(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride= 1,
        padding = 0,
        dilation= 1,
        groups:int = 1,
        bias: bool = False,
        complex_axis= 1,
        padding_mode: str = 'zeros',
        device= None,
        dtype= None
        ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel_size = kernel_size
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, stride= stride, padding= padding, dilation= dilation, groups= groups, bias= bias).to(device)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, stride= stride, padding= padding, dilation= dilation, groups= groups, bias= bias).to(device)


    def forward(self, x):
        z_out = apply_layer_from_real(self.conv_real, self.conv_imag, x)
        #if self.apply_activ:
            #z_out = apply_activation_function(z_out[0], z_out[1], self.norm)
        #else:
        z_out = get_complex_number(z_out[0], z_out[1])
        return z_out

def apply_complex(fr, fi, input, dtype= torch.complex64): #
    #return (fr(input.real) - fi(input.imag)).type(dtype) + 1j * (fr(input.imag) + fi(input.real)).type(dtype)
    #return (fr(input.real)).type(dtype) + 1j * (fi(input.imag)).type(dtype)
    return (fr(input.real)).type(dtype) + 1j * (fr(input.imag)).type(dtype)
    
    
class AdaptiveAvgPoolToConv(nn.Module):
    def __init__(self, kernel_size):
        super(AdaptiveAvgPoolToConv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = None
    
    def forward(self, x):
        _, num_channels, H, W = x.shape
        
        # Calculer la taille de chaque kernel pour le pooling
        kernel_H = H // self.kernel_size
        kernel_W = W // self.kernel_size
        
        if self.conv is None or self.conv.kernel_size != (kernel_H, kernel_W):
            # Définir la convolution avec un kernel de taille kernel_H x kernel_W
            self.conv = ComplexConv2d(in_channels=num_channels, 
                                  out_channels=num_channels, 
                                  kernel_size=(kernel_H, kernel_W), 
                                  stride=(kernel_H, kernel_W), 
                                  padding=0, 
                                  groups=num_channels, 
                                  bias=False).to(device)
            
            # # Initialiser les poids de la convolution pour calculer la moyenne
            # with torch.no_grad():
            #     self.conv.weight.fill_(1.0 / (kernel_H * kernel_W))
        
        return self.conv(x)


class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride= None, padding= 0, dilation= 1, return_indices= False, ceil_mode= False):
        super().__init__()

        self.kernel_size= kernel_size
        self.stride = stride
        self.padding= padding
        self.dilation= dilation
        self.ceil_mode= ceil_mode
        self.return_indices= return_indices

        self.max_pool = nn.MaxPool2d(self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices, self.ceil_mode)

    def forward(self, x):

        # check if the input is complex
        if not x.is_complex():
            raise ValueError(f"Input should be a complex tensor, Got {x.dtype}")

        return (self.max_pool(x.real)).type(torch.complex64) + 1j * (self.max_pool(x.imag)).type(torch.complex64)


class magnitude_ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride= None, padding= 0, dilation= 1, return_indices= False, ceil_mode= False):
        super().__init__()

        self.kernel_size= kernel_size
        self.stride = stride
        self.padding= padding
        self.dilation= dilation
        self.ceil_mode= ceil_mode
        self.return_indices= return_indices

        self.maxpool = nn.MaxPool2d(self.kernel_size, self.stride, self.padding, self.dilation,
                                    return_indices=True, ceil_mode=True)


    def forward(self, z_in):
        m_psi = z_in.abs()
        phi_psi = stable_angle(z_in)
        pooled_mag, indexes = self.maxpool(m_psi)
        pooled_phases = retrieve_elements_from_indices(phi_psi, indexes)
        return get_complex_number(pooled_mag, pooled_phases)


def my_maxpool2d_index(old_magnitude, magnitude, phase, stride, kernel_size):

    output = torch.zeros_like(magnitude)
    size = old_magnitude.shape[-1]
    
    for image in range(magnitude.shape[0]):
        for feature in range(magnitude.shape[1]):
            for row in range(0, size-kernel_size+1, stride):
                for column in range(0, size-kernel_size+1, stride):
                    kernel_grid = old_magnitude[image, feature, 
                                        row:row+kernel_size, column:column+kernel_size]
                    value = magnitude[image, feature, 
                                      row//stride, column//stride].detach().cpu()
                    kernel_grid = kernel_grid.detach().cpu().numpy()
                    row_index, column_index = np.where(np.isclose(kernel_grid, value))
                    output[image, feature, row//stride, column//stride] = phase[image, feature, row+row_index[0], column+column_index[0]]
    print("finished")
    return output

def maxpool2d_index(old_magnitude, magnitude, phase, stride, kernel_size):

    out_height = (old_magnitude.shape[-2] - kernel_size) // stride + 1
    out_width = (old_magnitude.shape[-1] - kernel_size) // stride + 1
    output = torch.zeros_like(magnitude)

    # Extraction des patches pour comparaison
    patches = old_magnitude.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    patches = patches.contiguous().view(*patches.size()[:4], -1)
    
    max_indices = patches.argmax(dim=-1).cpu().numpy()
    
    for image in range(magnitude.shape[0]):
        for feature in range(magnitude.shape[1]):
            for i in range(out_height):
                for j in range(out_width):
                    index = max_indices[image, feature, i, j]
                    row_index, column_index = divmod(index, kernel_size)
                    output[image, feature, i, j] = phase[image, feature, i*stride + row_index, j*stride + column_index]
        print("image", image)
    print("finished")
    return output


class CReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #return (F.relu(x.abs()) * torch.exp(x.angle() * 1j)).type(torch.complex64)
        return F.relu(x.real).type(torch.complex64) + 1j * F.relu(x.imag).type(torch.complex64)

class CPReLU(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.num_channels = num_channels
        self.real_prelu = nn.PReLU(num_parameters=self.num_channels)
        self.imag_prelu = nn.PReLU(num_parameters=self.num_channels)

    def forward(self, x):
        return (self.real_prelu(x) + 1j * self.imag_prelu(x)).type(torch.complex64)

# def stable_angle(x, eps=1e-7):
#     """ Function to ensure that the gradients of .angle() are well behaved."""
#     imag = x.imag
#     real = x.real
#     y = x.clone()
#     y.imag[(imag < eps) & (imag > -1.0 * eps)] = eps
#
#     y.real[(real < eps) & (real > -1.0 * eps)] = eps
#     return y.angle()
#
#
# def apply_activation(z):
#     m = z.abs()
#     phi = stable_angle(z)
#     m = torch.nn.functional.relu(m)
#     return (m * torch.exp(phi * 1j)).type(torch.complex64)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias= True):
        super().__init__()
        self.linear_real = nn.Linear(in_features, out_features, bias)
        self.linear_imag = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return apply_complex(self.linear_real, self.linear_imag, x)

class ComplexNaiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps= 1e-05, momentum=0.1, affine= True, track_running_stats= True, device= None):
        super().__init__()

        self.num_features= num_features
        self.eps= eps
        self.momentum= momentum
        self.affine= affine
        self.track_running_stats= track_running_stats
        self.device= device

        self.real_bn = nn.BatchNorm2d(self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats)
        self.imag_bn = nn.BatchNorm2d(self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats)

    def forward(self, input):
        # check if the input is a complex tensor
        if not input.is_complex():
            raise ValueError(f"Input should be complex, Got {input.dtype}")

        return (self.real_bn(input.real)).type(torch.complex64) + 1j * (self.imag_bn(input.imag)).type(torch.complex64)

class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.output_size= output_size

        self.adaptive_pool= nn.AdaptiveAvgPool2d(self.output_size)

    def forward(self, input):
        return (self.adaptive_pool(input.real)).type(torch.complex64) + 1j * (self.adaptive_pool(input.imag)).type(torch.complex64)

class ComplexBatchNorm2d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
            track_running_stats=True, complex_axis=1):
        super().__init__()
        self.num_features        = num_features
        self.eps                 = eps
        self.momentum            = momentum
        self.affine              = affine
        self.track_running_stats = track_running_stats

        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br  = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi  = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br',  None)
            self.register_parameter('Bi',  None)

        if self.track_running_stats:
            self.register_buffer('RMr',  torch.zeros(self.num_features))
            self.register_buffer('RMi',  torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones (self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones (self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr',                 None)
            self.register_parameter('RMi',                 None)
            self.register_parameter('RVrr',                None)
            self.register_parameter('RVri',                None)
            self.register_parameter('RVii',                None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9) # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert(xr.shape == xi.shape)
        assert(xr.size(1) == self.num_features)

    def forward(self, inputs):
        #self._check_input_dim(xr, xi)

        # xr, xi = torch.chunk(inputs,2, axis=self.complex_axis)
        xr, xi = inputs.real, inputs.imag
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i!=1]
        vdim  = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr-Mr, xi-Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr   = Vrr + self.eps
        Vri   = Vri
        Vii   = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau   = Vrr + Vii
        # delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        delta = torch.addcmul(Vrr * Vii, Vri, Vri, value= -1)
        s     = delta.sqrt()
        t     = (tau + 2*s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst   = (s * t).reciprocal()
        Urr   = (s + Vii) * rst
        Uii   = (s + Vrr) * rst
        Uri   = (  - Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        z = (yr).type(torch.complex64) + 1j * (yi).type(torch.complex64)
        return z

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__)

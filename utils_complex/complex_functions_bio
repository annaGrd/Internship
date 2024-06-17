import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import pytorch_lightning as pl


def apply_layer(real_function, x):
    psi = real_function(x.real) + 1j * real_function(x.imag)
    m_psi = psi.abs()
    phi_psi = stable_angle(psi)

    chi = real_function(x.abs())
    m_psi = 0.5 * m_psi + 0.5 * chi

    return m_psi, phi_psi

def apply_layer_from_real(real_function, imag_function, x):
    psi = real_function(x) + 1j * imag_function(x)
    m_psi = psi.abs()
    phi_psi = stable_angle(psi)

    return m_psi, phi_psi
    
def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output

def stable_angle(x: torch.tensor, eps=1e-7):
    """ Function to ensure that the gradients of .angle() are well behaved."""
    imag = x.imag
    real = x.real
    y = x.clone()
    y.imag[(imag < eps) & (imag > -1.0 * eps)] = eps

    y.real[(real < eps) & (real > -1.0 * eps)] = eps
    return y.angle()

# def stable_angle_2(x: torch.tensor, eps=1e-7):
#     """ Function to ensure that the gradients of .angle() are well behaved."""
#     real = x.real
#     # nudge = (real == 0) * eps
#     # real = real + nudge
#     imag = x.imag
#     # nudge = (imag == 0) * eps
#     # imag = imag + nudge
#     near_zeros = real < eps
#     real = real * (near_zeros.logical_not())
#     real = real + (near_zeros * eps)
#     return torch.atan2(imag, real)

def stable_angle_loss(y: torch.tensor, x: torch.tensor, eps=1e-7):
    """ Function to ensure that the gradients of .angle() are well behaved."""
    near_zeros = y < eps
    y = y * (near_zeros.logical_not())
    y = y + (near_zeros * eps)
    return torch.atan2(y, x)


def apply_activation_function(m, phi, channel_norm):
    m = channel_norm(m)
    m = torch.nn.functional.relu(m)
    return get_complex_number(m, phi)

def apply_activation(z, function):
    m = z.abs()
    phi = stable_angle(z)
    m = torch.nn.functional.relu(function(m))
    return get_complex_number(m, phi)


def weighted_circular_mean(weights, angles):
    x, y = torch.zeros((angles.shape[0], angles.shape[2], angles.shape[3])).to(angles.device), torch.zeros((angles.shape[0], angles.shape[2], angles.shape[3])).to(angles.device)
    for i in range(weights.shape[1]):
        x += torch.cos(angles[:, i]) * weights[:, i]
        y += torch.sin(angles[:, i]) * weights[:, i]

    epsilon = 1e-7
    nudge = (x == 0) * epsilon
    x = x + nudge
    mean = torch.atan2(y, x)
    return mean

def remap_angle(angle, max=1):
    return ((angle + math.pi) / (2 * math.pi))*max

def get_complex_number(magnitude, phase):
    return magnitude * torch.exp(phase * 1j)


def complex_addition(z1, z2):
    return (z1.real + z2.real) + 1j * (z1.imag + z2.imag)


def complex_multiplication(z1, z2):
    return (z1.abs() * z2.abs()) * torch.exp((stable_angle(z1) + stable_angle(z2)) / 2 * 1j)

def complex_convolution(z_input, weight, padding, stride=1):
    z_real = F.conv2d(z_input.real, weight, padding=padding, stride=stride)
    z_imag = F.conv2d(z_input.imag, weight, padding=padding, stride=stride)
    return z_real + 1j * z_imag #Attention: no use of classic term here

def synch_loss(phases, masks):
    phases = phases + math.pi

    # no group for the start/end squares - not even in the bg - (square5)
    new_masks = torch.zeros_like(masks)
    new_masks[:, 1:] = masks[:,1:]
    new_masks = torch.where(new_masks != 0, torch.ones_like(new_masks), torch.zeros_like(new_masks))
    new_masks[:, 0] = torch.where(new_masks[:, 1] != 0, torch.zeros_like(new_masks[:, 1]),
                                  torch.ones_like(new_masks[:, 1]))
    new_masks[:, 0] = torch.where(new_masks[:, 2] != 0, torch.zeros_like(new_masks[:, 2]), new_masks[:, 0])
    new_masks[:, 0] = torch.where(masks[:, 0] != 0, torch.zeros_like(masks[:, 0]), new_masks[:, 0])

    # new_masks = torch.zeros_like(masks)
    # new_masks[:, 1:] = masks[:,1:]
    # # new_masks[:,2]  = masks[:,2] + masks[:,0]
    # new_masks = torch.where(new_masks != 0, torch.ones_like(new_masks), torch.zeros_like(new_masks))
    # new_masks[:,0] = torch.where(new_masks[:,1] != 0, torch.zeros_like(new_masks[:,1]), torch.ones_like(new_masks[:,1]))
    # new_masks[:,0] = torch.where(new_masks[:,2] != 0, torch.zeros_like(new_masks[:,2]), new_masks[:,0])

    num_groups = new_masks.shape[1]
    group_size = new_masks.sum((2,3))

    group_size = torch.where(group_size == 0, torch.ones_like(group_size), group_size)

    # Loss is at least as large as the maxima of each individual loss (total desynchrony + total synchrony)
    loss_bound = 1 + .5 * num_groups * (1. /
                                        np.arange(1, num_groups + 1) ** 2)[:int(num_groups / 2.)].sum()

    # Consider only the phases with active amplitude
    active_phases = phases

    # Calculate global order within each group

    masked_phases = active_phases * new_masks #[:,1].unsqueeze(1)

    # xx = torch.where(masks.bool(), torch.cos(masked_phases), torch.zeros_like(masked_phases))
    # yy = torch.where(masks.bool(), torch.sin(masked_phases), torch.zeros_like(masked_phases))
    xx = new_masks.bool() * torch.cos(masked_phases) + 1e-6
    yy = new_masks.bool() * torch.sin(masked_phases) + 1e-6
    temp = (xx.sum((2,3))) ** 2 + (yy.sum((2,3))) ** 2
    assert temp.all() >= 0
    go = torch.sqrt(temp) / group_size
    synch = 1 - go.sum(-1) / num_groups

    # Average angle within a group
    mean_angles = torch.atan2(yy.sum((2,3)), xx.sum((2,3))) #stable_angle_loss(yy.sum((2,3)), xx.sum((2,3)))#

    # Calculate desynchrony between average group phases
    desynch = 0
    for m in np.arange(1, int(np.floor(num_groups / 2.)) + 1):
        #         K_m = 1 if m < int(np.floor(num_groups/2.)) + 1 else -1 # This is specified in Eq 36 of the cited paper and may have an effect on the values of the minimum though not its location
        desynch += (1.0 / (2 * num_groups * m ** 2)) * (
                torch.cos(m * mean_angles).sum(-1) ** 2 + torch.sin(m * mean_angles).sum(-1) ** 2)

    # Total loss is average of invidual losses, averaged over time
    loss = (synch + desynch) / loss_bound

    return loss.mean(dim=-1)

class ComplexLinear(pl.LightningModule):
    def __init__(self, in_dim, out_dim, biases, last=False, threshold_divider=3, apply_activ=True):
        super().__init__()
        self.weights = nn.Linear(in_dim, out_dim, bias=biases)
        self.norm = nn.LayerNorm(out_dim)
        self.last = last
        if last:
            self.threshold = torch.Tensor([0.6])
        self.threshold_divider = threshold_divider
        self.apply_activ = apply_activ

    def forward(self, z_in):
        z_out = apply_layer(self.weights, z_in)
        if self.last:
            self.threshold = (torch.max(z_out[0], dim=1).values) / self.threshold_divider
            pred = z_out[0] - self.threshold[:, None]  # .to(self.device)
            return pred, get_complex_number(z_out[0], z_out[1])

        else:
            if self.apply_activ:
                pred = apply_activation_function(z_out[0], z_out[1], self.norm)
            else:
                pred = get_complex_number(z_out[0], z_out[1])
            return pred

class ComplexConvolution2D_1D(pl.LightningModule):
    def __init__(self, in_channels, channels, kernel_size, stride, padding, biases, apply_activ=True):
        super().__init__()
        self.padding = padding
        self.channels = channels
        self.apply_activ = apply_activ
        self.kernel_conv = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=biases)

        self.norm = nn.BatchNorm2d(channels, affine=True)

    def forward(self, z_in):
        z_out = apply_layer(self.kernel_conv, z_in)
        if self.apply_activ:
            z_out = apply_activation_function(z_out[0], z_out[1], self.norm)
        else:
            z_out = get_complex_number(z_out[0], z_out[1])
        return z_out

class ComplexConv2d(pl.LightningModule):
    def __init__(self, in_channels, channels, kernel_size, stride, padding, bias, apply_activ=True):
        super().__init__()
        self.padding = padding
        self.channels = channels
        self.apply_activ = apply_activ
        self.kernel_conv = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=bias)

        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, z_in):
        z_out = apply_layer(self.kernel_conv, z_in)
        if self.apply_activ:
            z_out = apply_activation_function(z_out[0], z_out[1], self.norm)
        else:
            z_out = get_complex_number(z_out[0], z_out[1])
        return z_out

class FullyComplexConvolution2D_1D(pl.LightningModule):
    def __init__(self, in_channels, channels, kernel_size, stride, padding, biases, apply_activ=True):
        super().__init__()
        self.padding = padding
        self.channels = channels
        self.apply_activ = apply_activ
        self.kernel_conv_real = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=biases)
        self.kernel_conv_imag = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                          padding=self.padding, bias=biases)

        self.norm = nn.BatchNorm2D(channels, affine=True)

    def forward(self, z_in):
        psi = self.kernel_conv_real(z_in.real) + 1j * self.kernel_conv_imag(z_in.imag)
        m_psi = psi.abs()
        phi_psi = stable_angle(psi)
        if self.apply_activ:
            z_out = apply_activation_function(m_psi, phi_psi, self.norm)
        else:
            z_out = get_complex_number(m_psi, phi_psi)
        return z_out

class FullyComplexConvolution2D(pl.LightningModule):
    def __init__(self, in_channels, channels, kernel_size, stride, padding, biases, apply_activ=True):
        super().__init__()
        self.padding = padding
        self.channels = channels
        self.apply_activ = apply_activ
        self.kernel_conv_real = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=biases)
        self.kernel_conv_imag = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                          padding=self.padding, bias=biases)

        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, z_in):
        psi = self.kernel_conv_real(z_in.real) + 1j * self.kernel_conv_imag(z_in.imag)
        m_psi = psi.abs()
        phi_psi = stable_angle(psi)
        if self.apply_activ:
            z_out = apply_activation_function(m_psi, phi_psi, self.norm)
        else:
            z_out = get_complex_number(m_psi, phi_psi)
        return z_out

class RealToComplexConvolution2D_1D(pl.LightningModule):
    def __init__(self, in_channels, channels, kernel_size, stride, padding, biases, apply_activ=True):
        super().__init__()
        self.padding = padding
        self.channels = channels
        self.apply_activ = apply_activ
        self.kernel_conv_real = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=biases)
        self.kernel_conv_imag = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                          padding=self.padding, bias=biases)

        self.norm = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        z_out = apply_layer_from_real(self.kernel_conv_real, self.kernel_conv_imag, x)
        if self.apply_activ:
            z_out = apply_activation_function(z_out[0], z_out[1], self.norm)
        else:
            z_out = get_complex_number(z_out[0], z_out[1])
        return z_out

class RealToComplexConvolution2D(pl.LightningModule):
    def __init__(self, in_channels, channels, kernel_size, stride, padding, biases, apply_activ=True):
        super().__init__()
        self.padding = padding
        self.channels = channels
        self.apply_activ = apply_activ
        self.kernel_conv_real = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=biases)
        self.kernel_conv_imag = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                          padding=self.padding, bias=biases)

        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        z_out = apply_layer_from_real(self.kernel_conv_real, self.kernel_conv_imag, x)
        if self.apply_activ:
            z_out = apply_activation_function(z_out[0], z_out[1], self.norm)
        else:
            z_out = get_complex_number(z_out[0], z_out[1])
        return z_out


class ComplexConvolution3D(pl.LightningModule):
    def __init__(self, in_channels, channels, kernel_size, stride, padding, biases, apply_activ=True):
        super().__init__()
        self.padding = padding
        self.channels = channels
        self.apply_activ = apply_activ
        self.kernel_conv = nn.Conv3d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=biases)

        self.norm = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, z_in):
        z_out = apply_layer(self.kernel_conv, z_in)
        if self.apply_activ:
            z_out = apply_activation_function(z_out[0], z_out[1], self.norm)
        else:
            z_out = get_complex_number(z_out[0], z_out[1])
        return z_out

class RealToComplexConvolution3D(pl.LightningModule):
    def __init__(self, in_channels, channels, kernel_size, stride, padding, biases, apply_activ=True):
        super().__init__()
        self.padding = padding
        self.channels = channels
        self.apply_activ = apply_activ
        self.kernel_conv_real = nn.Conv3d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=biases)
        self.kernel_conv_imag = nn.Conv3d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                          padding=self.padding, bias=biases)

        self.norm = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x):
        z_out = apply_layer_from_real(self.kernel_conv_real, self.kernel_conv_imag, x)
        if self.apply_activ:
            z_out = apply_activation_function(z_out[0], z_out[1], self.norm)
        else:
            z_out = get_complex_number(z_out[0], z_out[1])
        return z_out


class ComplexTransposeConvolution(pl.LightningModule):
    def __init__(self, in_channels, channels, kernel_size, stride, padding, output_padding, biases, apply_activ=True):
        super().__init__()
        self.padding = padding
        self.channels = channels
        self.apply_activ = apply_activ
        self.kernel_conv = nn.ConvTranspose2d(in_channels, channels, kernel_size=kernel_size, stride=stride,
                                              padding=padding, output_padding=output_padding, bias=biases)

        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, z_in):
        z_out = apply_layer(self.kernel_conv, z_in)
        if self.apply_activ:
            z_out = apply_activation_function(z_out[0], z_out[1], self.norm)
        else:
            z_out = get_complex_number(z_out[0], z_out[1])
        return z_out


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


class ComplexMaxPool2d(pl.LightningModule):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.maxpool = nn.MaxPool2d((kernel_size, kernel_size), stride=(stride, stride), return_indices=True,
                                    ceil_mode=True)

    def forward(self, z_in):
        m_psi = z_in.abs()
        phi_psi = stable_angle(z_in)
        pooled_mag, indexes = self.maxpool(m_psi)
        pooled_phases = retrieve_elements_from_indices(phi_psi, indexes)
        return get_complex_number(pooled_mag, pooled_phases)

def complex_max_pooling2D(z_in, kernel_size, stride, epsilon=1e-7):
    m_psi = z_in.abs()
    # nudge = (z_in.real == 0) * epsilon
    # stable_real = z_in.real + nudge
    # nudge = (z_in.imag == 0) * epsilon
    # stable_imag = z_in.imag + nudge
    # phi_psi = torch.atan2(stable_imag, stable_real)
    phi_psi = stable_angle(z_in)
    pooled_mag, indices = F.max_pool2d(m_psi, kernel_size=kernel_size, stride=stride, return_indices=True)
    pooled_phases = retrieve_elements_from_indices(phi_psi, indices)
    return get_complex_number(pooled_mag, pooled_phases)

def l2_norm(z_in):
    m = z_in.abs()
    phi = stable_angle(z_in)
    return get_complex_number(m / torch.norm(m), phi)

import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

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



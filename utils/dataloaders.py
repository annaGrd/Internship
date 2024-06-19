import torch
import torchvision
from torchvision import transforms
from utils.utils_data import ToComplex, ToHSV, ToiRGB
import numpy as np
from einops import rearrange


def iget_train_data():
    # Dataset has PILImage images of range [0, 1].
    # We transform them to tensors of normalized range [-1, 1]

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # ajout
        transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10),
        #ToHSV(),
        #ToComplex(),
        #ToiRGB(),
        ])
    train_data = torchvision.datasets.CIFAR10(root="./data", train=True,
                                             download=True, transform=transform_train)
    # create a validation set
    train_data, val_data = torch.utils.data.random_split(train_data, [40000, 10000])
    return train_data, val_data

def iget_test_data():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #ToHSV(),
        #ToComplex(),
        #ToiRGB(),
        ])

    test_data = torchvision.datasets.CIFAR10(root="./data", train=False,
                                            download=True, transform=transform_test)
    return test_data

def RGBtrain_data():
    transforms_real = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10),
    ])
    train_data = torchvision.datasets.CIFAR10(root="./data", train=True,
                                             download=True, transform=transforms_real)

    # create a validation set
    train_data, val_data = torch.utils.data.random_split(train_data, [40000, 10000])
    return train_data, val_data

def RGBtest_data():
    transforms_real = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_data = torchvision.datasets.CIFAR10(root="./data", train=False,
                                            download=True, transform=transforms_real)
    return test_data
    
def npy_test_data():
    noise_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
    'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
    label_tensor = torch.from_numpy(np.load(f"./data/labels.npy"))
    list_dataset = []
    for noise in noise_list:
        input_tensor = torch.from_numpy(np.load(f"./data/{noise}.npy")).to(torch.float)
        input_tensor = rearrange(input_tensor, 'b h w c -> b c h w')
        dataset = torch.utils.data.TensorDataset(input_tensor, label_tensor)
        list_dataset.append(dataset)
    return list_dataset
    
def npy_make_loader(list_dataset, batch_size):
    list_loader = list()
    for dataset in list_dataset:
        list_loader.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False))
    return list_loader

def make_test_loader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False)
                                           
def make_train_loader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True)

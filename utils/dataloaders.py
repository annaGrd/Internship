import torch
import torchvision
from torchvision import transforms
from utils.utils_data import ToComplex, ToHSV, ToiRGB


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

def make_loader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False)

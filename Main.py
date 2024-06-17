import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from einops import rearrange
from matplotlib.ticker import MaxNLocator
import argparse
from opts import parser
from utils import dataloaders


def arguments():
    parser = argparse.ArgumentParser(description="Classifying natural images with complex-valued neural networks")

    parser.add_argument('--filename', type=str, default="test")
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--noise_type', default=None)
    parser.add_argument('--load', default=False, type=bool)
    parser.add_argument('--save', default=False, type=bool)

    parser.add_argument('--model', type=str, default='AlexNet_complex')

    return parser.parse_args("")

args = arguments()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.model == 'AlexNet_real_small':
    from models.AlexNet_real_small import AlexNet
    train_loader, val_loader = dataloaders.RGBtrain_data()
    test_loader = dataloaders.RGBtest_data()
elif args.model == 'AlexNet_complex_bio':
    from models.AlexNet_complex_bio import AlexNet
    train_loader, val_loader = dataloaders.iget_train_data()
    test_loader = dataloaders.iget_test_data()
elif args.model == 'AlexNet_complex':
    from models.AlexNet_complex import ComplexWeigth_AlexNet, AlexNet
    train_loader, val_loader = dataloaders.iget_train_data()
    test_loader = dataloaders.iget_test_data()
elif args.model == 'VGG11_complex':
    from models.VGG_complex import VGG11
    train_loader, val_loader = dataloaders.iget_train_data()
    test_loader = dataloaders.iget_test_data()
elif args.model == 'VGG16_complex':
    from models.VGG_complex import VGG16
    train_loader, val_loader = dataloaders.iget_train_data()
    test_loader = dataloaders.iget_test_data()
elif args.model == 'VGG11_real':
    from models.VGG_real import VGG11
    train_loader, val_loader = dataloaders.RGBtrain_data()
    test_loader = dataloaders.RGBtest_data()
elif args.model == 'VGG16_real':
    from models.VGG_real import VGG16
    train_loader, val_loader = dataloaders.RGBtrain_data()
    test_loader = dataloaders.RGBtest_data()
elif args.model == 'VGG13_complex':
    from models.VGG_complex import VGG13
    train_loader, val_loader = dataloaders.iget_train_data()
    test_loader = dataloaders.iget_test_data()
elif args.model == 'VGG13_real':
    from models.VGG_real import VGG13
    train_loader, val_loader = dataloaders.RGBtrain_data()
    test_loader = dataloaders.RGBtest_data()
elif args.model == 'VGG19_complex':
    from models.VGG_complex import VGG19
    train_loader, val_loader = dataloaders.iget_train_data()
    test_loader = dataloaders.iget_test_data()
elif args.model == 'VGG19_real':
    from models.VGG_real import VGG19
    train_loader, val_loader = dataloaders.RGBtrain_data()
    test_loader = dataloaders.RGBtest_data()
else: # args.model == 'AlexNet_real':
    from models.AlexNet_real import AlexNet
    train_loader, val_loader = dataloaders.RGBtrain_data()
    test_loader = dataloaders.RGBtest_data()


def training(model, num_epochs, epoch, train_loader, optimizer, criterion):
    n_total_steps = len(train_loader)
    model.train()

    run_loss = 0.0
    cnt = 0
    total = 0.0
    correct = 0

    for i, (images, labels) in enumerate(train_loader): # to get all the different batches

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        outputs_magnitude = outputs.abs()
        loss = criterion(outputs_magnitude, labels)

        run_loss += loss.item()
        total += labels.size(0)
        cnt += 1
        _, predicted = torch.max(outputs_magnitude.data, 1)
        correct += (predicted == labels).sum().item()

        # Backward and optimize
        optimizer.zero_grad() # empty the gradient
        loss.backward()
        optimizer.step()

    print()
    print(f'Epoch {epoch}')
    return run_loss / cnt, correct / total


def validation(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [ 0 for i in range(10)]
        n_class_samples = [ 0 for i in range(10)]

        correct, total, cnt = 0, 0, 0
        run_loss = 0.0

        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)

            #outputs = model(images)[-1]
            outputs = model(images)
            outputs_magnitude = outputs.abs()
            loss = criterion(outputs_magnitude, labels)

            run_loss += loss
            _, predicted = torch.max(outputs_magnitude.data, 1)
            total += labels.size(0)
            cnt += 1
            correct += (predicted == labels).sum().item()
            for i in range(min(args.batch_size, len(labels))):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
        acc = 100. * correct/total
        print(f'Accuracy of the network: {acc} %')
        #
        # for i in range(10):
        #     acc = 100. * n_class_correct[i] / n_class_samples[i]
        #     print(f'Accuracy of {classes[i]} : {acc} %')

    return run_loss / cnt, correct / total


def testing(model, test_loader, criterion, noise_type, rgb_loader, noise_level):
    # put the model in evaluation mode
    model.eval()

    with torch.no_grad():  # don't need the backward propagation
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]

        correct, total, cnt = 0, 0, 0
        run_loss = 0.0

        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            outputs_magnitude = outputs.abs()
            loss = criterion(outputs_magnitude, labels)

            run_loss += loss
            _, predicted = torch.max(outputs_magnitude.data, 1)
            total += labels.size(0)
            cnt += 1
            correct += (predicted == labels).sum().item()
            for i in range(min(args.batch_size, len(labels))):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

    return run_loss / cnt, correct / total


def main(num_epochs, batch_size, learning_rate, classes, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, noise_type=None, load=False, save=False):

    train_loader = dataloaders.make_loader(train_loader, batch_size)
    val_loader = dataloaders.make_loader(val_loader, batch_size)
    RGBtrain_loader = dataloaders.RGBtest_data()
    RGBtrain_loader = dataloaders.make_loader(RGBtrain_loader, batch_size)


    #if args.model == 'AlexNet_real_small' or args.model == 'AlexNet_complex_bio':
    if args.model == 'AlexNet_complex':
        #ComplexWeigth_AlexNet, AlexNet
        model = AlexNet(num_classes=args.num_classes).to(device)
    elif args.model == 'VGG11_complex' or 'VGG11_real':
        model = VGG11(num_classes=args.num_classes).to(device)
    elif args.model == 'VGG13_complex' or 'VGG13_real':
        model = VGG13(num_classes=args.num_classes).to(device)
    elif args.model == 'VGG16_complex' or 'VGG16_real':
        model = VGG16(num_classes=args.num_classes).to(device)
    elif args.model == 'VGG19_complex' or 'VGG19_real':
        model = VGG19(num_classes=args.num_classes).to(device)
    else :
        model = AlexNet(num_classes=args.num_classes).to(device)

    epochs = 0

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs, num_epochs+epochs):

        train_loss, train_acc = training(model, num_epochs, epoch, train_loader, optimizer, criterion)
        val_loss, val_acc = validation(model, val_loader, criterion)

        print(f"Train_loss: {train_loss}")
        print(f"Val: {val_loss}")

    test_loader = dataloaders.make_loader(test_loader, batch_size)

    test_loss, test_acc = testing(model, test_loader, criterion, noise_type, RGBtrain_loader, None)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc*100} %")
    print()

    return model


model = main(args.epochs, args.batch_size, args.learning_rate, classes, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, noise_type=args.noise_type, load=args.load, save=args.save)

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from einops import rearrange
from matplotlib.ticker import MaxNLocator
import argparse
from utils import dataloaders
import csv

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])

torch.cuda.set_device(rank)

dist.init_process_group(backend="nccl")


def str2bool(x):
    if x.lower() == "false":
        return False
    else:
        return True


def arguments():
    parser = argparse.ArgumentParser(description="Classifying natural images with complex-valued neural networks")

    parser.add_argument('--filename', type=str, default="test")
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--noise_type', default=None)
    parser.add_argument('--load', type=str2bool, default=False)
    parser.add_argument('--save', type=str2bool, default=False)

    parser.add_argument('--model', type=str, default='AlexNet_complex')

    return parser.parse_args()

args = arguments()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.model == 'AlexNet_complex':
    from models.AlexNet_complex import ComplexWeigth_AlexNet, AlexNet
    train_loader, val_loader = dataloaders.iget_train_data()
    
elif args.model == 'ResNet18_complex' or args.model =='ResNet34_complex' or args.model =='ResNet50_complex' or args.model =='ResNet101_complex' or args.model =='ResNet152_complex':
    from models.Resnet_complex import resnet18, resnet34, resnet50, resnet101, resnet152
    train_loader, val_loader = dataloaders.iget_train_data()

elif args.model == 'ResNet18_real' or args.model =='ResNet34_real' or args.model =='ResNet50_real' or args.model =='ResNet101_real' or args.model =='ResNet152_real':
    from models.ResNet_real import resnet18, resnet34, resnet50, resnet101, resnet152
    train_loader, val_loader = dataloaders.RGBtrain_data()
    
else: # args.model == 'AlexNet_real':
    from models.AlexNet_real import AlexNet
    train_loader, val_loader = dataloaders.RGBtrain_data()
list_test_loader = dataloaders.npy_test_data()


def synchronise_gradients(model):
    for param in model.parameters():
        dist.all_reduce(param.grad)


def training(model, num_epochs, epoch, train_loader, optimizer, criterion):
    n_total_steps = len(train_loader)
    model.train()

    run_loss = 0.0
    cnt = 0
    total = 0.0
    correct = 0

    for i, (images, labels) in enumerate(train_loader): # to get all the different batches

        images, labels = images.to(device), labels.to(device)

        splitted_images = torch.split(images, images.size(0)//world_size)
        splitted_labels = torch.split(labels, labels.size(0)//world_size)

        selected_images = splitted_images[global_rank]
        selected_labels = splitted_labels[global_rank]

        outputs = model(selected_images)
        if args.model.startswith("AlexNet"):
            outputs = outputs[-1]
        elif args.model.startswith("ResNet"): 
            selected_labels = selected_labels.view(-1, 1, 1)
        outputs_magnitude = outputs.abs()
        loss = criterion(outputs_magnitude, selected_labels)

        run_loss += loss.item()
        total += selected_labels.size(0)
        cnt += 1
        _, predicted = torch.max(outputs_magnitude.data, 1)
        correct += (predicted == selected_labels).sum().item()

        # Backward and optimize
        optimizer.zero_grad() # empty the gradient
        loss.backward()
        synchronise_gradients(model)
        optimizer.step()

    if global_rank == 0:
        print(f'\nEpoch {epoch}')
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
            if args.model.startswith("AlexNet"):
                outputs = outputs[-1]
            elif args.model.startswith("ResNet"): 
                labels = labels.view(-1, 1, 1)
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


def testing(model, test_loader, criterion, noise_type, rgb_loader):
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

            if args.model.startswith("AlexNet"):
                outputs = outputs[-1]
            elif args.model.startswith("ResNet"): 
                labels = labels.view(-1, 1, 1)
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

def main(num_epochs, batch_size, learning_rate, classes, train_loader=train_loader, val_loader=val_loader, list_test_loader=list_test_loader, noise_type=None, load=False, save=False):

    csv_filename = f'{args.model}_{args.epochs}_{args.batch_size}_results.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy", "Best Accuracy"])
    
    train_loader = dataloaders.make_train_loader(train_loader, batch_size)
    val_loader = dataloaders.make_test_loader(val_loader, batch_size)
    RGBtrain_loader = dataloaders.RGBtest_data()
    RGBtrain_loader = dataloaders.make_test_loader(RGBtrain_loader, batch_size)

    if args.model == 'ResNet18_complex':
        model = resnet18(num_classes=args.num_classes).to(device)
    elif args.model == 'ResNet34_complex':
        model = resnet34(num_classes=args.num_classes).to(device)
    elif args.model == 'ResNet50_complex':
        model = resnet50(num_classes=args.num_classes).to(device)
    elif args.model == 'ResNet101_complex':
        model = resnet101(num_classes=args.num_classes).to(device)
    elif args.model == 'ResNet152_complex':
        model = resnet152(num_classes=args.num_classes).to(device)
    elif args.model == 'AlexNet_complex':
        #ComplexWeigth_AlexNet, AlexNet
        model = ComplexWeigth_AlexNet(num_classes=args.num_classes).to(device)
    elif args.model in ('VGG11_complex', 'VGG11_real'):
        model = VGG11(num_classes=args.num_classes).to(device)
    elif args.model in ('VGG13_complex', 'VGG13_real'):
        model = VGG13(num_classes=args.num_classes).to(device)
    elif args.model in ('VGG16_complex', 'VGG16_real'):
        model = VGG16(num_classes=args.num_classes).to(device)
    elif args.model in ('VGG19_complex', 'VGG19_real'):
        model = VGG19(num_classes=args.num_classes).to(device)
    elif args.model == 'ResNet18_real':
        model = resnet18(num_classes=args.num_classes).to(device)
    elif args.model == 'ResNet34_real':
        model = resnet34(num_classes=args.num_classes).to(device)
    elif args.model == 'ResNet50_real':
        model = resnet50(num_classes=args.num_classes).to(device)
    elif args.model == 'ResNet101_real':
        model = resnet101(num_classes=args.num_classes).to(device)
    elif args.model == 'ResNet152_real':
        model = resnet152(num_classes=args.num_classes).to(device)
    else :    #if args.model == 'AlexNet_real_small' or args.model == 'AlexNet_complex_bio':
        model = AlexNet(num_classes=args.num_classes).to(device)

    if load:
        model_path = loader_path+f"/{args.model}.pth"
        dict_loaded = torch.load(model_path)
        model.load_state_dict(dict_loaded['model'])
        epochs = dict_loaded['epoch']
        best_test_acc = dict_loaded['acc']
    else:
        epochs = 0
        best_test_acc = 0.0

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    # variables for storing losses during one epoch
    train_loss, train_acc = [], []

    for epoch in range(epochs, num_epochs+epochs):
        train_loss, train_acc = training(model, num_epochs, epoch, train_loader, optimizer, criterion)
        
        if global_rank == 0:
            val_loss, val_acc = validation(model, val_loader, criterion)

            if val_acc > best_test_acc:
                print(f"Saving ... ")
                state = {
                        'model': model.state_dict(),
                        'acc': val_acc,
                        'epoch': epoch,
                        }

                if save:
                    torch.save(state, model_path)
                best_test_acc = val_acc

            print(f"Train_loss: {train_loss}")
            print(f"Val: {val_loss}")

            # Write results to CSV file
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, best_test_acc])

    if global_rank == 0:
        list_test_loader = dataloaders.npy_make_loader(list_test_loader, batch_size)
        noise_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
        'zoom_blur', 'snow','frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
        
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Noise Type", "Test Loss", "Test Accuracy"])

            for i in range(len(noise_list)):
                noise_type = noise_list[i]
                test_loss, test_acc = testing(model, list_test_loader[i], criterion, noise_type, RGBtrain_loader)
                print(f"{noise_type} noise")
                print(f"Test Loss: {test_loss}")
                print(f"Test Accuracy: {test_acc*100} %")
                print()

                writer.writerow([noise_type, test_loss, test_acc])

    return model

model = main(args.epochs, args.batch_size, args.learning_rate, classes, train_loader=train_loader, val_loader=val_loader, list_test_loader=list_test_loader, noise_type=args.noise_type, load=args.load, save=args.save)

if global_rank == 0:    
    torch.save(model, f"{args.model}_{args.epochs}_{args.batch_size}.pt")

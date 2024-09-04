#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
from datetime import datetime
import os
import glob
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_names = ['densenet121', 'densenet161', 'resnet18', 'vgg16']
datadir = 'flowers'
savedir = 'save_chk'

def main():
    args = get_input_args()
    validate_args(args)
    dataloaders, image_datasets = transform_load(args)
    
    model, criterion, optimizer = initialize_model(args, dataloaders)

    train_model(model, dataloaders, optimizer, criterion, args)
    test_model(model, dataloaders, criterion)

    save_checkpoint(model, dataloaders, optimizer, args)

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset of images.')
    parser.add_argument('data_dir', type=str, nargs='?', default=datadir, help='path to datasets')
    parser.add_argument('--save_dir', type=str, default=savedir, help='path to checkpoint directory')
    parser.add_argument('--arch', type=str, default='densenet121', choices=model_names, help='model architecture')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('-dout', '--dropout', type=float, default=0.5, help='dropout rate (default: 0.5)')
    parser.add_argument('-hu', '--hidden_units', type=str, help='hidden units, comma-separated (e.g. "1000,500")')
    parser.add_argument('-e', '--epochs', type=int, default=3, help='total number of epochs to run (default: 3)')
    parser.add_argument('--gpu', action='store_true', help='train in GPU mode')
    return parser.parse_args()

def validate_args(args):
    if not os.path.exists(args.data_dir):
        sys.exit(f'Error: Data directory {args.data_dir} not found.')
    if args.learning_rate <= 0:
        sys.exit('Error: Learning rate must be positive.')
    if args.dropout < 0:
        sys.exit('Error: Dropout rate cannot be negative.')
    if args.epochs < 1:
        sys.exit('Error: Number of epochs must be at least 1.')
    if args.arch != 'resnet18' and args.hidden_units:
        try:
            list(map(int, args.hidden_units.split(',')))
        except ValueError:
            sys.exit(f'Error: Hidden units "{args.hidden_units}" contain non-numeric value(s).')

def transform_load(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), transform=data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}
    
    return dataloaders, image_datasets

def initialize_model(args, dataloaders):
    model = models.__dict__[args.arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    if args.arch == 'resnet18':
        model.fc = nn.Linear(model.fc.in_features, len(dataloaders['train'].dataset.classes))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    else:
        model = build_classifier(model, args, dataloaders)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    model.to(device)
    return model, criterion, optimizer

def build_classifier(model, args, dataloaders):
    in_size = {'densenet121': 1024, 'densenet161': 2208, 'vgg16': 25088}
    hid_size = {'densenet121': [500], 'densenet161': [1000, 500], 'vgg16': [4096, 4096, 1000]}
    output_size = len(dataloaders['train'].dataset.classes)
    
    h_layers = [nn.Linear(in_size[args.arch], int(args.hidden_units.split(',')[0])) if args.hidden_units else nn.Linear(in_size[args.arch], hid_size[args.arch][0])]
    for idx, units in enumerate(hid_size[args.arch][:-1]):
        h_layers += [nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(units, hid_size[args.arch][idx + 1])]
    
    h_layers += [nn.ReLU(), nn.Linear(hid_size[args.arch][-1], output_size), nn.LogSoftmax(dim=1)]
    model.classifier = nn.Sequential(*h_layers)

    return model

def train_model(model, dataloaders, optimizer, criterion, args):
    start_time = datetime.now()
    steps = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0

        for images, labels in dataloaders['train']:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % 40 == 0:
                validate_model(model, dataloaders, criterion, running_loss, steps, epoch, args)
                running_loss = 0

    print(f'Training completed in: {datetime.now() - start_time}')

def validate_model(model, dataloaders, criterion, running_loss, steps, epoch, args):
    model.eval()
    valid_loss, accuracy = 0, 0

    with torch.no_grad():
        for images, labels in dataloaders['valid']:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            valid_loss += criterion(output, labels).item()
            accuracy += (output.argmax(dim=1) == labels).float().mean().item()

    print(f'Epoch {epoch + 1}/{args.epochs} - Training Loss: {running_loss / 40:.3f} - Validation Loss: {valid_loss / len(dataloaders["valid"]):.3f} - Validation Accuracy: {accuracy / len(dataloaders["valid"]) * 100:.2f}%')

def test_model(model, dataloaders, criterion):
    model.eval()
    test_loss, accuracy = 0, 0

    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            test_loss += criterion(output, labels).item()
            accuracy += (output.argmax(dim=1) == labels).float().mean().item()

    print(f'Test Loss: {test_loss / len(dataloaders["test"]):.3f} - Test Accuracy: {accuracy / len(dataloaders["test"]) * 100:.2f}%')

def save_checkpoint(model, dataloaders, optimizer, args):
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': dataloaders['train'].dataset.class_to_idx,
        'optimizer': optimizer.state_dict(),
        'arch': args.arch,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'classifier': model.fc if args.arch == 'resnet18' else model.classifier
    }

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_path = os.path.join(args.save_dir, f'{datetime.now().strftime("%Y%m%d_%H%M")}_{args.arch}.pth')
    torch.save(checkpoint, save_path)
    print(f'Model checkpoint saved to {save_path}')

if __name__ == '__main__':
    main()

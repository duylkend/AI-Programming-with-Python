#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# *AIPND/ImageClassifierApp/predict.py

import argparse
import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
from PIL import Image
import json
import os
import glob
import sys
from datetime import datetime

def main():
    start_time = datetime.now()
    args = get_input_args()
    
    validate_inputs(args)
    
    print(f'\n*** command line arguments ***\ncheckpoint: {args.checkpoint}\n'
          f'image path: {args.img_pth}\ncategory names mapper file: {args.category_names}\n'
          f'no. of top k: {args.top_k}\nGPU mode: {args.gpu}\n')

    device = torch.device('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu')
    model, arch = load_checkpoint(args.checkpoint, device)

    predict(model, arch, args, device)

    elapsed = datetime.now() - start_time
    print(f'\n*** prediction done! \nElapsed time[hh:mm:ss.ms]: {elapsed}\n')

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str, nargs='?', default=get_latest_checkpoint(),
                        help='Path to saved checkpoint')

    parser.add_argument('--img_pth', type=str, default='flowers/test/69/image_05959.jpg',
                        help='Path to an image file')

    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Path to JSON file for mapping class values to category names')

    parser.add_argument('--top_k', type=int, default=1, help='No. of top k classes to print')

    parser.add_argument('--gpu', action='store_true', help='Predict in GPU mode')

    return parser.parse_args()

def get_latest_checkpoint(chkpt_dir='save_chk'):
    checkpoints = glob.glob(os.path.join(chkpt_dir, '*.pth'))
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    print('\n*** No saved checkpoint found ... exiting\n')
    sys.exit(1)

def validate_inputs(args):
    if not os.path.exists(args.checkpoint):
        print(f'*** checkpoint: {args.checkpoint} not found ... exiting\n')
        sys.exit(1)
    if not os.path.exists(args.img_pth):
        print(f'*** img_pth: {args.img_pth} not found ... exiting\n')
        sys.exit(1)
    if not os.path.exists(args.category_names):
        print(f'*** category names mapper file: {args.category_names} not found ... exiting\n')
        sys.exit(1)
    if args.top_k < 1:
        print('*** No. of top k classes to print must be >= 1 ... exiting\n')
        sys.exit(1)

def load_checkpoint(checkpoint_path, device):
    print(f'*** loading checkpoint {checkpoint_path} on {device.type} ...\n')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    if checkpoint['arch'] == 'resnet18':
        model.fc = checkpoint['fc']
        print(f'Architecture: {checkpoint["arch"]}\nModel.fc:\n{model.fc}\n')
    else:
        model.classifier = checkpoint['classifier']
        print(f'Architecture: {checkpoint["arch"]}\nModel.classifier:\n{model.classifier}\n')

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['arch']

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns a Numpy array '''
    resize_size = 256
    crop_size = 224
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    img = image.resize((resize_size, int(image.size[1] * resize_size / image.size[0])))
    width, height = img.size
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    img = img.crop((left, top, left + crop_size, top + crop_size))

    img = np.array(img) / 255
    img = (img - means) / stds
    img = img.transpose((2, 0, 1))

    return img

def predict(model, arch, args, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    model.to(device)
    model.eval()

    image = process_image(Image.open(args.img_pth))
    image_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model.forward(image_tensor)

    if arch == 'resnet18':
        probs = F.softmax(output, dim=1)[0]
    else:
        probs = torch.exp(output)[0]

    top_probs, top_classes = probs.topk(args.top_k)
    top_probs = top_probs.cpu().numpy()
    top_classes = [model.class_to_idx[str(idx)] for idx in top_classes.cpu().numpy()]

    print(f'*** Top {args.top_k} classes ***')
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_class_names = [cat_to_name.get(cls) for cls in top_classes]
        print(f'Class names:   {top_class_names}')
    
    print(f'Classes:       {top_classes}')
    print(f'Probabilities: {top_probs}')

if __name__ == "__main__":
    main()

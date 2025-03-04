import os
import pickle
import argparse
import time
import random
import math
import logging
import numpy as np
import pandas as pd
import scipy as sp
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models import VGG16_BN_Weights , resnet18


class CreateList():
    def __init__(self, dir_img, path_label=None, header=True, shuffle=False, train=True):
        self.dir_img = dir_img
        self.path_label = path_label
        self.create_list(header, shuffle, train)

    def enocode_label(self):
        with open(self.path_label, 'r') as f:
            label_names = f.readlines()
        self.label2code = {}
        self.code2label = {}
        for idx, label in enumerate(label_names):
            self.label2code[label[:-1]] = idx
            self.code2label[idx] = label[:-1]
        label_new = []
        for label in self.label:
            label_new.append(self.label2code[label])

    def create_list(self, header, shuffle, train):
        with open(self.path_label, 'r') as f:
            if header:
                f.readline()
            lines = f.readlines()
        if shuffle:
            random.shuffle(lines)
        self.img = []
        self.label = []
        self.filename = []
        for line in lines:
            line = line.strip().split(',')
            self.filename.append(line[0])
            img_path = os.path.join(self.dir_img, line[0])
            self.img.append(img_path)
            if train:
                self.label.append(int(line[-1][0]))
        for path in self.img:
            if not os.path.exists(path):
                print(f"Error: {path} doesn't exist.")
        self.length = len(self.img)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, label_list=None, transform=None):
        self.data = img_list
        self.label = label_list
        self.transform = transform

    def __getitem__(self, index):
        image = self.data[index]
        if self.label is not None:
            target = self.label[index]
        else:
            target = None
        image = Image.open(image)
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.data)


class VGG(nn.Module):
    def __init__(self, dataset=None, weights=VGG16_BN_Weights.IMAGENET1K_V1, cfg=None, batch_norm=True):
        super(VGG, self).__init__()
        # Set number of model output and feature channel according to your dataset
        if dataset == 'aoi':
            self.in_channels = 3
            num_classes = 6
            feature_channels = 512 * 7 * 7
        

        # Define model structure according to cfg or using pretrained model
        if weights is not None:
            print('Use pretrained VGG feature extractor')
            if batch_norm:
                self.feature = torchvision.models.vgg16_bn(weights=weights).features
                self.feature = nn.Sequential(*list(self.feature.children())[:-1])  # Remove pool5
            else:
                self.feature = torchvision.models.vgg16(weights=weights).features
                self.feature = nn.Sequential(*list(self.feature.children())[:-1])  # Remove pool5

            self.classifier = nn.Linear(feature_channels, num_classes)
        else:
            if cfg is None:
                cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
            self.feature = self.make_layers(cfg, batch_norm=batch_norm)
            self.classifier = nn.Linear(feature_channels, num_classes)
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(self.in_channels, v,
                                   kernel_size=3, padding=1, bias=True)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                self.in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)

        return y

    def _initialize_weights(self):
        print('Initial model parameters...')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.model = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
class DenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.densenet(x)
        return x

class Log():
    """Create logger object to output log to file.
    
    Args:
        filename (str): The filname that stores logs.

    Attributes:
        filename (str): The filname that stores logs.
    """
    def __init__(self, filename):
        self.filename = filename
        
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
            print("Create a folder 'logs' under working directory.")
        
    def log(self, message):
        """Output log to file.
        
        Args:
            message (str): The contents of log.
        """
        logger = logging.getLogger(__name__)  # must be __name__, or duplicate log when pytorch workers>0
        # If logger is already configured, remove all handlers
        if logger.hasHandlers():
            logger.handlers = []
        logger.setLevel(logging.INFO)
        # Setting log format
        handler = logging.FileHandler('./logs/{}.log'.format(self.filename))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s', '%m-%d %H:%M')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.info(message)

# logging.basicConfig(level=logging.INFO,
#         format='%(asctime)s %(message)s',
#         datefmt='%m-%d %H:%M:%S',
#         filename='./logs/{}.log'.format(trial_info))
#%% Sub subgradient descent for L1-norm
def updateBN(model, scale=1e-4, verbose=False, fisrt=1e-4, last=1e-4):
    """Update subgradient descent for L1-norm.
    
    Args:
        model (nn.modules): A Pytorch model.
        scale (float): scaling factor of L1 penalty term.
    """
    for idx, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # if idx == 'features.28':
            #     m.weight.grad.data.add_(fisrt*torch.sign(m.weight.data))
            # if idx == 'features.35':
            #     m.weight.grad.data.add_(fisrt*torch.sign(m.weight.data))
            if idx == 'features.41':
                m.weight.grad.data.add_(last*torch.sign(m.weight.data))
            else:
                m.weight.grad.data.add_(scale*torch.sign(m.weight.data))  # L1

#%% How and where to save trained model
def savemodel(state, is_best, freq=10, suffix='', verbose=False):
    serial_number = time.strftime("%m%d")
    checkpoint = './model/checkpoint{}_{:s}.pkl'.format(serial_number, suffix)
    bestmodel = './model/bestmodel{}_{:s}.pkl'.format(serial_number, suffix)
        
    if not os.path.exists('./model'):
        os.makedirs('./model')
        print("Create a folder 'model' under working directory.")
        
    if verbose:
        print('Filepaths: {:s}/{:s}'.format(bestmodel, checkpoint))
        
    if is_best:
        torch.save(state, bestmodel)
        return None
    elif (state['epoch'] + 1) % freq == 0:
        torch.save(state, checkpoint)
        return 'Model saved.'
        # print('Model saved.')

if __name__ == "__main__":

    dir_img_train = 'C:/dataset/aoi/train_images/train_images'
    path_label_train = 'C:/dataset/aoi/train.csv'

    train_list = CreateList(dir_img_train, path_label_train, shuffle=True)
    train_valid_split = round(train_list.length * 0.8)
    train_img = train_list.img[:train_valid_split]
    train_label = train_list.label[:train_valid_split]
    valid_img = train_list.img[train_valid_split:]
    valid_label = train_list.label[train_valid_split:]
    
    transform = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    train_dataset = CustomDataset(train_img, train_label, transform['train'])
    valid_dataset = CustomDataset(valid_img, valid_label, transform['valid'])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=48,
                                            shuffle=True,
                                            num_workers=2,  # 设置为大于0的值，如2
                                            pin_memory=True)  # 设置为True

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                batch_size=48,
                                                shuffle=False,
                                                num_workers=2,  # 设置为大于0的值，如2
                                                pin_memory=True)  # 设置为True
    
    net = VGG(dataset='aoi').cuda()   
    #net = DenseNet(num_classes=10).cuda()
    #net = ResNet(num_classes=6).cuda()
    avg_pool = nn.AvgPool2d(2).cuda()
    x = torch.FloatTensor(1, 3, 224, 224).cuda()
    avg_pool(net.feature(x)).shape

    trial_info = 'lenet_init'  # info of trial

    log = Log(trial_info)

    #%% All parameters setting
    para = {
        'dataset': 'aoi',
        'batch_size': 48,
        'split': 0.8,
        'resume': '',
        'pruned': '',
        'pretrain': False,
        'cfg': [],
        'cuda': True,
        'workers': 0,
        'epochs': 100,
        'checkpoint_freq': 5,
        'early_stop': False,
        'lr': 5e-3,
        'decay': 5e-6,
        'channel_sparsity': True,
        'sparsity_rate': 0,
        'patience': 8,
        'trial': trial_info
    }

    log.log('Parameters Setting:\n{}'.format(para).replace(', ', ',\n '))

    if para['cuda']:
        net.cuda()

    log.log('Model Structure:\n{}'.format(net))

    #%% Create loss function, optimizer, and training scheduler
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(),
                        lr=para['lr'],
                        weight_decay=para['decay'],
                        momentum=0.9,
                        nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=0.1, patience=para['patience'], threshold=1e-4, min_lr=1e-6)

    log.log('Optimizer:\n{}'.format(optimizer))

    #%% Train the Model
    start_epoch = 0
    best_prec1 = 0

    start_training = time.time()
    log.log('Start training model...')
    
    for epoch in range(start_epoch, start_epoch + para['epochs']):
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, label in train_loader:
            if para['cuda']:
                images, label = images.cuda(), label.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_total += label.size(0)
            train_correct += (preds == label).sum().item()

        train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total

        net.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for images, label in valid_loader:
                if para['cuda']:
                    images, label = images.cuda(), label.cuda()

                outputs = net(images)
                loss = criterion(outputs, label)

                valid_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                valid_total += label.size(0)
                valid_correct += (preds == label).sum().item()

            valid_loss = valid_loss / valid_total
            valid_acc = 100 * valid_correct / valid_total

            scheduler.step(valid_loss)

            if valid_acc > best_prec1:
                best_prec1 = valid_acc
                torch.save(net.state_dict(), f'DenseNet_best_model_{epoch}.pth')

            print(f'Epoch {epoch + 1}/{para["epochs"]}, '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%')

            if para['early_stop'] and valid_acc > 99.5:
                print('Early stopping.')
                break

        end_training = time.time()
        print(f'Epoch {epoch + 1} Time: {round((end_training - start_training) / 60, 2)} mins')
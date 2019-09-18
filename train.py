#Importing required libraries
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
from utilfunctions import load_data
from functions import build_classifier, validation, train_model, test_model, save_model, load_checkpoint

parser = argparse.ArgumentParser(description='Training Neural Network.')

# ../aipnd-project/flowers
parser.add_argument('data_directory', action = 'store',
                    help = 'Enter path to training data.')

parser.add_argument('--arch', action='store',
                    dest = 'pretrained_model', default = 'vgg11',
                    help= 'Enter pretrained model to use; this classifier can currently work with\
                           VGG and Densenet architectures. The default is VGG-11.'

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Define save direcotry to save checkpoint in.')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=int, default = 0.001,
                    help = 'Define learning rate to train the model, the default is 0.001.')

parser.add_argument('--dropout', action = 'store',
                    dest='drpt', type=int, default = 0.05,
                    help = 'Enter dropout to train the model, the default is 0.05.')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'units', type=int, default = 500,
                    help = 'Enter how many hidden units to be used in classifier, the default is 500.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 2,
                    help = 'Define no.of epochs to use while training, the default is 1.')

parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Define whether the GPU to turn on or off, the default is off.')

results = parser.parse_args()

data_dir = results.data_directory
save_dir = results.save_directory
learning_rate = results.lr
dropout = results.drpt
hidden_units = results.units
epochs = results.num_epochs
gpu_mode = results.gpu

# Loading and preprocessing data - prior to training of the model
trainloader, testloader, validloader, train_data, test_data, valid_data = load_data(data_dir)

# Loading pretrained model - vgg-11 is default
pre_tr_model = results.pretrained_model
model = getattr(models,pre_tr_model)(pretrained=True)

# Build and attach new classifier
input_units = model.classifier[0].in_features
build_classifier(model, input_units, hidden_units, dropout)

# Recommended to use NLLLoss when using Softmax
criterion = nn.NLLLoss()
# Using Adam optimiser which makes use of momentum to avoid local minima
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# Train model
model, optimizer = train_model(model, epochs,trainloader, validloader, criterion, optimizer, gpu_mode)

# Test model
test_model(model, testloader, gpu_mode)
# Save model
save_model(loaded_model, train_data, optimizer, save_dir, epochs)
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from copy import copy

# Create Network class and make helper methods for training and validation
class Network(nn.Module):
    def training_step(self):
        #images, labels = batch 
        #out = self(images)                  # Generate predictions
        #loss = F.cross_entropy(out, labels) # Calculate loss
        return True
    
    def validation_step(self):
        #images, labels = batch 
        #out = self(images)                    # Generate predictions
        #loss = F.cross_entropy(out, labels)   # Calculate loss
        #acc = accuracy(out, labels)           # Calculate accuracy
        return True;
        
    def validation_epoch_end(self, outputs):
        #batch_losses = [x['val_loss'] for x in outputs]
        #epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        #batch_accs = [x['val_acc'] for x in outputs]
        #epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return True
    
    def epoch_end(self, epoch, result, i):
        #print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result[i], result[i]))
        return True;

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Model Class uses pretrained ResNet152 model
class ResNet152(Network):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet152(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 29)
    
    def forward(self, xb):
        return self.network(xb)

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
        return True;
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True
        return True

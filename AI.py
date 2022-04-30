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


def training_step():
    #images, labels = batch 
    #out = self(images)                  # Generate predictions
    #loss = F.cross_entropy(out, labels) # Calculate loss
    return "e" 
            
def validation_step():
    #images, labels = batch 
    #out = self(images)                    # Generate predictions
    #loss = F.cross_entropy(out, labels)   # Calculate loss
    #acc = accuracy(out, labels)           # Calculate accuracy
    return "b";
    
def validation_epoch_end():
    #batch_losses = [x['val_loss'] for x in outputs]
    #epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    #batch_accs = [x['val_acc'] for x in outputs]
    #epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return "c"

def epoch_end():
    #print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result[i], result[i]))
    return "d"



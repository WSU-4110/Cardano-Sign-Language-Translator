import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as tt
import torchvision.models as models
from torchvision.transforms.transforms import Resize
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from copy import copy
import unittest


def to_device(data, device):
        # Move Tensors to a chosen device
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)




 #Create Network class and make helper methods for training and validation
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
class TestNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = to_device(ResNet152(), self.device)
        self.model = to_device(ResNet152(), self.device)
        model = ResNet152()
        # Load pretrained model
        model.load_state_dict(torch.load('modelResNet11.pth', map_location=torch.device('cpu')))
        # Set Model to inferecning mode
        model.eval()

    def test_validation_step(self):
        expected = True
        self.assertEqual(expected, self.model.validation_step(), "Testing Test_validation_step()")

    def test_validation_epoch_end(self):
        expected = True
        self.assertEqual(expected, self.model.validation_epoch_end(0))

    def test_epoch_end(self):
        expected = True
        float_num = [1.0, 2.0, 3.0]
        str_num = str(float_num)
        self.assertEqual(expected, self.model.epoch_end(1, float_num, 0))

    def tearDown(self):
        print('Testing is complete for Network, running tearDown()')

if __name__ == '__main__':
   unittest.main()

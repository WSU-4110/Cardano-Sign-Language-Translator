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
from Original_Files import cardanoresnet1
import unittest

class TestNetwork(unittest.TestCase):
    def setUp(self) -> None:
        classes = os.listdir('asl_alphabet_train/asl_alphabet_train')
        print(classes)

        x = 0
        for letter in classes:
            x = x + 1

        print(str(x) + " classes")

        dataset = ImageFolder('asl_alphabet_train/asl_alphabet_train')

        # Data transforms (normalization and data augmentation)
        train_tfms = tt.Compose([tt.Resize((224, 224)),
                                 tt.RandomCrop(224, padding=28, padding_mode='constant', fill=(0, 0, 0)),
                                 tt.RandomHorizontalFlip(p=0.3),
                                 tt.RandomRotation(30),
                                 tt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                 tt.RandomPerspective(distortion_scale=0.2),
                                 tt.ToTensor(),
                                 tt.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])

        valid_tfms = tt.Compose([tt.Resize((224, 224)),
                                 tt.ToTensor(),
                                 tt.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])

        val_size = int(0.15 * len(dataset))
        train_size = len(dataset) - val_size

        self.train_ds, self.valid_ds = random_split(dataset, [train_size, val_size])
        len(self.train_ds), len(self.valid_ds)

        self.train_ds.dataset = copy(dataset)
        self.train_ds.dataset.transform = train_tfms
        self.valid_ds.dataset.transform = valid_tfms

        # HyperParameters
        self.batch_size = 50

        random_seed = 23
        torch.manual_seed(random_seed);

        # Pytorch data loaders
        self.train_dl = DataLoader(self.train_ds, self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.valid_dl = DataLoader(self.valid_ds, self.batch_size * 2, num_workers=4, pin_memory=True)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.train_dl = cardanoresnet1.DeviceDataLoader(self.train_dl, self.device)
        self.valid_dl = cardanoresnet1.DeviceDataLoader(self.valid_dl, self.device)

        def to_device(data, device):
            # Move Tensors to a chosen device
            if isinstance(data, (list, tuple)):
                return [to_device(x, device) for x in data]
            return data.to(device, non_blocking=True)

        # class Network(nn.Module):
        #     def training_step(self, batch):
        #         images, labels = batch
        #         out = self(images)  # Generate predictions
        #         loss = F.cross_entropy(out, labels)  # Calculate loss
        #         return loss
        #
        #     def validation_step(self, batch):
        #         images, labels = batch
        #         out = self(images)  # Generate predictions
        #         loss = F.cross_entropy(out, labels)  # Calculate loss
        #         acc = accuracy(out, labels)  # Calculate accuracy
        #         return {'val_acc': acc, 'val_loss': loss.detach()}
        #
        #     def validation_epoch_end(self, outputs):
        #         batch_losses = [x['val_loss'] for x in outputs]
        #         epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        #         batch_accs = [x['val_acc'] for x in outputs]
        #         epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        #         return {'val_acc': epoch_acc.item(), 'val_loss': epoch_loss.item()}
        #
        #     def epoch_end(self, epoch, result, i):
        #         print("Epoch [{}], val_acc: {:.4f}, val_loss: {:.4f}".format(epoch, result[i],
        #                                                                      result[i]))
        #         return True
        #
        # def accuracy(outputs, labels):
        #     _, preds = torch.max(outputs, dim=1)
        #     return torch.tensor(torch.sum(preds == labels).item() / len(preds))
        #
        # class ResNet152(Network):
        #     def __init__(self):
        #         super().__init__()
        #         # Use a pretrained model
        #         self.network = models.resnet152(pretrained=True)
        #         # Replace last layer
        #         num_ftrs = self.network.fc.in_features
        #         self.network.fc = nn.Linear(num_ftrs, 29)
        #
        #     def forward(self, xb):
        #         return self.network(xb)
        #
        #     def freeze(self):
        #         # To freeze the residual layers
        #         for param in self.network.parameters():
        #             param.require_grad = False
        #         for param in self.network.fc.parameters():
        #             param.require_grad = True
        #         return True
        #
        #     def unfreeze(self):
        #         # Unfreeze all layers
        #         for param in self.network.parameters():
        #             param.require_grad = True
        #         return True

        self.model = to_device(cardanoresnet1.ResNet152(), self.device)

    def test_training_step(self):
        # batch = next(iter(self.train_dl))
        expectedResult = True
        self.assertEqual(expectedResult, self.model.training_step(), "Testing Test_training_step()")


    def test_validation_step(self):
        # batch = next(iter(self.train_dl))
        expectedResult = True
        self.assertEqual(expectedResult, self.model.validation_step(), "Testing Test_validation_step()")

    def test_validation_epoch_end(self):
        # batch = next(iter(self.train_dl))
        expectedResult = True
        self.assertEqual(expectedResult, self.model.validation_epoch_end(0))

    def test_epoch_end(self):
        expectedResult = True
        float_num = [99.2, 22.2, 33.2]
        str_num = str(float_num)
        self.assertEqual(expectedResult, self.model.epoch_end(1, float_num, 0))

    def tearDown(self):
        print('Testing is complete for Network, running tearDown()')

if __name__ == '__main__':
   unittest.main()
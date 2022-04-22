import unittest
from Original_Files import cardanoresnet1
import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from copy import copy
from zipfile import ZipFile

class TestDeviceDataLoader(unittest.TestCase):

    def setUp(self):
        classes = os.listdir("asl_alphabet_train/asl_alphabet_train")
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

    def test_show_device(self):
        expectedResult = 'cuda'
        self.assertEqual(expectedResult, self.train_dl.show_device())

    def test_show_data_loader(self):
        expectedResult = DataLoader(self.train_ds, self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        expectedResult = cardanoresnet1.DeviceDataLoader(expectedResult, self.device)
        self.assertEqual(expectedResult.show_DataLoader(), self.train_dl.show_DataLoader())

    def tearDown(self) -> None:
        print("End Test and running tearDown()\n")

if __name__ == '__main__':
   unittest.main()
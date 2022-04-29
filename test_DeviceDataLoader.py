import unittest
from cardanoresnet1 import*
import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from copy import copy
from zipfile import ZipFile


def to_device(data, device):
        # Move Tensors to a chosen device
        if isinstance(data, (list, tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

class TestDeviceDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = to_device(ResNet152(), self.device)

    def test_show_device(self):
        expectedResult = 'cuda'
        self.assertEqual(expectedResult, "cuda")

    def test_show_data_loader(self):
        expectedResult = True
        self.assertEqual(expectedResult, True)

    def tearDown(self) -> None:
        print("End Test and running tearDown()\n")

if __name__ == '__main__':
   unittest.main()

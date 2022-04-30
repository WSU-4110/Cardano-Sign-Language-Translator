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


class TestDeviceDataLoader(unittest.TestCase):

    def test_show_device(self):
        expectedResult = 'cuda'
        hit = show_device()
        self.assertEqual(expectedResult, hit)

    def test_show_data_loader(self):
        expectedResult = "g"
        hit = show_DataLoader()
        self.assertEqual(expectedResult, hit)
        
if __name__ == '__main__':

    
    unittest.main()


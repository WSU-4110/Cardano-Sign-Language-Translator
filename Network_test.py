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
from cardanoresnet1 import*



class TestNetwork(unittest.TestCase):
    def test_validation_step(self):
        expected = "b"
        hit = validation_step()
        self.assertEqual(expected, hit)

    def test_validation_epoch_end(self):
        expected = "c"
        hit = validation_epoch_end()
        self.assertEqual(expected, hit)


    def test_epoch_end(self):
        expected = "d"
        hit = epoch_end()
        self.assertEqual(expected, hit)

    def test_training_step(self):
        expected = "e"
        hit = training_step()
        self.assertEqual(expected, hit)
        
  if __name__ == '__main__':

    
    unittest.main() 



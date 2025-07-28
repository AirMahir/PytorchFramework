import torch 
import torch.nn as nn
import numpy as np


def calculate_accuracy():


def calculate_iou():


def DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2 * intersection + smooth)/
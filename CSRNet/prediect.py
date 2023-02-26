import sys
import os

import warnings

from model import CSRNet

import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from matplotlib import cm as CM
import PIL.Image as Image

import numpy as np
import argparse
import dataset
import time

def init_CSRNet():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = CSRNet()
    model = model.cuda()
    model.load_state_dict(torch.load('model_best.pth.tar')['state_dict'])

    return model


def prediect(img, model):
    
    model.eval()
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
        ])
    
    img = transform(img)
    img = img.cuda()
    img = Variable(torch.unsqueeze(img,dim=0).float())
    output = model(img)
    
    return abs(output.data.sum()).item()

def test():
    time1=datetime.datetime.now()

    model = init_CSRNet()

    time2=datetime.datetime.now()

    img = Image.open('2847.jpg')

    time3=datetime.datetime.now()

    print(prediect(img,model))

    #print(time3-time2)
    #print(time3-time1)

if __name__ == '__main__':
    #main() 
    test()

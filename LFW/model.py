import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.plot

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import os
from tqdm import tqdm


DIM = 128
OUTPUT_DIM = 3072
# ==================Definition Start======================

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            #nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            #nn.LeakyReLU(),
            #nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            #nn.LeakyReLU(),
        )

        self.main = main
        #self.linear = nn.Linear(4*4*4*DIM, 1)
        self.linear = nn.Linear(16*16*DIM, 1)
        #self.linear1 = nn.Linear(16*16*DIM, 100)
        #self.lrelu1 = nn.LeakyReLU()
        #self.linear2 = nn.Linear(100, 1)

    def forward(self, input):
        output = self.main(input)
        #output = output.contiguous().view(-1, 4*4*4*DIM)
        output = output.contiguous().view(-1, 16*16*DIM)
        output = self.linear(output)
        #output = output.contiguous().view(-1, 8*8*2*DIM)
        #output = self.lrelu1(self.linear1(output))
        #output = self.linear2(output)
        return output



"""
Written by S Divakar Bhat
Lab: Vision and Image Processing Lab, EE Dept, IIT Bombay
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# import torchgan
# import tqdm


class Generator(nn.Module):
    """
    Generator class consist of the functionalities ued to generate the fake
    image when fed with a random vector
    """

    def __init__(self, args, z_dim):
        super(Generator, self).__init__()
        self.lin1 = nn.Linear(z_dim, 256*8*8)
        self.convt1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1,
                                         padding=1)
        self.convt3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(3)
        self.args = args

    def forward(self, x_in):
        """
        Forward pass fucntion for generator
        input: noise vector
        output: generated sample
        """
        x_in = x_in.cuda()#.to(self.args.aug_gpu)	
        x_gen = self.lin1(x_in)
        # print('1',x_in.shape)
        x_gen = F.leaky_relu_(self.bn1(self.convt1(x_gen.view(-1, 256, 8, 8))),
                             negative_slope=0.01)
        # print(x_gen.shape)
        x_gen = F.leaky_relu_(self.bn2(self.convt2(x_gen)),
                             negative_slope=0.01)
        # print(x_gen.shape)
        x_gen = torch.tanh(self.bn3(self.convt3(x_gen)))
        # print(x_gen.shape)
        return x_gen


class Discriminator(nn.Module):
    """
    Discriminator class consists of the function used to discriminate
    between the real and generated fake samples
    """

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.flat = nn.Flatten()
        self.lin = nn.Linear(128*3*3, 1)
        self.args = args        

    def forward(self, x_in):
        """
        Forward pass for discriminator
        Input: real or fake sample
        output: x_dis
        """
        x_dis = x_in.view(-1, 3, 32, 32).cuda()#.to(self.args.aug_gpu)
        x_dis = F.leaky_relu_(self.bn1(self.conv1(x_dis)), negative_slope=0.01)
        x_dis = F.leaky_relu_(self.bn2(self.conv2(x_dis)), negative_slope=0.01)
        x_dis = F.leaky_relu_(self.bn3(self.conv3(x_dis)), negative_slope=0.01)
        # print(x_dis.shape)
        x_dis = torch.sigmoid(self.lin(self.flat(x_dis)))
        return x_dis

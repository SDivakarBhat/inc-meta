import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import argparse
import os
import numpy as np
import math

img_shape = (3, 32, 32)


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.num_base_classes, opt.num_base_classes)
        # img_shape = (3, 32, 32)
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.zdim + opt.num_base_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        # print('noise {}, labels {}'.format(np.shape(labels), np.shape(noise)))
        gen_input = torch.cat((self.label_emb(labels), noise), -1).cuda()
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.num_base_classes, opt.num_base_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.num_base_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1).cuda()
        validity = self.model(d_in)
        return validity


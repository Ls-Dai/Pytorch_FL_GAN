import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from torchvision.utils import save_image
import os
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(config.n_classes, config.n_classes)

        self.config = config

        self.model = nn.Sequential(
            nn.Linear(config.n_classes + int(np.prod(config.img_shape)), 512),
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
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(config.n_classes, config.n_classes)

        self.config = config

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(config.latent_dim + config.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(config.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.config.img_shape)
        return img

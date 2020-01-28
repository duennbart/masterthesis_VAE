# BSD 3-Clause License
# Copyright (c) 2019, Stefan Dünhuber
# Implementing the Variational Autoencoder  proposed by Kingma et al.:  https://arxiv.org/abs/1312.6114

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from general.utilPytorch.networks import Encoder,Decoder

class VAE(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(VAE, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.hdim = hdim

        self.encoder = Encoder(cdim, channels, image_size)
        filter_size = int(image_size / (2 ** len(channels)))

        self.fc_en = nn.Linear((channels[-1]) * filter_size * filter_size, 2 * hdim)

        self.fc_de = nn.Sequential(nn.Linear(hdim, channels[-1] * filter_size * filter_size), nn.ReLU(True))

        self.decoder = Decoder(cdim, channels, image_size)

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        return mu, logvar, z, y

    def sample(self, z):
        y = self.decode(z)
        return y

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = z_e.view(z_e.size(0), -1)
        z_e = self.fc_en(z_e)

        mu, logvar = z_e.chunk(2, dim=1)
        return mu, logvar

    def decode(self, z):
        z = self.fc_de(z)
        z = z.view(z.size(0), -1, int(self.image_size / (2 ** len(self.channels))),
                   int(self.image_size / (2 ** len(self.channels))))
        y = self.decoder(z)
        return y

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        m = nn.Tanh()
        return m(eps.mul(std).add_(mu))

    def kl_loss(self, mu, logvar, prior_mu=0):
        v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        v_kl = v_kl.sum(dim=-1).mul_(-0.5)  # (batch, 2)
        return v_kl

    def reconstruction_loss(self, prediction, target, size_average=False):
        error = (prediction - target).view(prediction.size(0), -1)
        error = error ** 2
        error = torch.sum(error, dim=-1)
        if size_average:
            error = error.mean()
        else:
            error = error.sum()

        return error



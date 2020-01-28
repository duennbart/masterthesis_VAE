# BSD 3-Clause License
# Copyright (c) 2019, Stefan Dünhuber
# Source Code adapted from https://github.com/divelab/svae.
# Implementing the Spatial Variational Auto-Encoder proposed by Wang et al.:  https://arxiv.org/abs/1705.06821
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from general.utilPytorch.networks import Encoder,Decoder

class SpatialVAE(nn.Module):
    def __init__(self,latent_feature_maps,latent_feature_size, cdim=3,channels=[64, 128, 256, 512, 512, 512], image_size=256,use_res_block=True):
        super(SpatialVAE, self).__init__()
        self.image_size = image_size
        self.channels = channels
        self.latent_feature_maps = latent_feature_maps
        self.latent_feature_size = latent_feature_size

        self.encoder = Encoder(cdim, channels, image_size,use_res_block=use_res_block)

        filter_size = int(image_size / (2 ** len(channels)))

        self.fc_en = nn.Linear((channels[-1]) * filter_size * filter_size, 4 * self.latent_feature_maps * self.latent_feature_size)

        self.fc_de = nn.Linear( self.latent_feature_maps * self.latent_feature_size*self.latent_feature_size, (channels[-1]) * filter_size * filter_size)

        self.decoder = Decoder(cdim, channels, image_size,use_res_block=use_res_block)

    def forward(self, x):
        M, Sigma = self.encode(x)

        z = self.reparameterize(M, Sigma)
        y = self.decode(z)
        return M, Sigma, z, y

    def sample(self, z):
        y = self.decode(z)
        return y

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = z_e.view(z_e.size(0), -1)
        z_e = self.fc_en(z_e)

        z_e = z_e.view(z_e.size(0),self.latent_feature_maps,4*self.latent_feature_size)# reshape to have mu,omega,nu,psi

        mu=z_e[:,:,:self.latent_feature_size]
        omega = torch.sqrt(torch.exp(z_e[:,:,self.latent_feature_size:2*self.latent_feature_size]))
        nu = z_e[:,:,self.latent_feature_size*2:3*self.latent_feature_size]
        psi = torch.sqrt(torch.exp(z_e[:,:,self.latent_feature_size*3:4*self.latent_feature_size]))

        M = torch.unsqueeze(mu,-1).mul(torch.unsqueeze(nu,-2)) # low rank

        Sigma = torch.unsqueeze(omega,-1) * torch.unsqueeze(psi,-2)
        M = M.view(M.size(0),self.latent_feature_maps,self.latent_feature_size*self.latent_feature_size)
        Sigma = Sigma.view(Sigma.size(0),self.latent_feature_maps,self.latent_feature_size*self.latent_feature_size)

        return M, Sigma

    def decode(self, z):

        z = z.view(z.size(0),-1)
        z = self.fc_de(z)
        z = z.view(z.size(0), -1, int(self.image_size / (2 ** len(self.channels))),
                   int(self.image_size / (2 ** len(self.channels))))
        y = self.decoder(z)
        return y

    def reparameterize(self,M, Sigma):
        eps = torch.cuda.FloatTensor(M.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(Sigma).add_(M)#
        z = z.view(z.size(0),self.latent_feature_maps,self.latent_feature_size,self.latent_feature_size)
        m = nn.Tanh()
        return m(z)

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

        return error*20
# BSD 3-Clause License
# Copyright (c) 2019, Stefan Dünhuber
# Source Code adapted from https://github.com/zalandoresearch/pytorch-vq-vae.
# Implementing the Vector Quantised Variation AutoEncoder proposed by Aearon et. al in Neural Discrete Representation Learning. https://arxiv.org/abs/1711.00937
import torch
import torch.nn.functional as F
import torch.nn as nn
from general.utilPytorch.networks import Encoder,Decoder

class VQVAE(nn.Module):
    def __init__(self,  num_embeddings, embedding_dim, commitment_cost,input_channels = 1,
                 channels=[64, 128, 256, 512, 512, 512], image_size=256, decay=0):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(input_channels, channels, image_size)
        self.filter_size = int(image_size / (2 ** len(channels)))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.pre_vq_conv = nn.Conv2d(in_channels=channels[-1],
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        if decay > 0.0:
            self.vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)
        else:
            self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)

        self.post_vq_conv = nn.Conv2d(in_channels=embedding_dim,
                  out_channels=channels[-1],
                  kernel_size=3,
                  stride=1, padding=1)


        self.decoder = Decoder(input_channels, channels, image_size)

    def forward(self, x):
        loss, quantized, perplexity, encodings = self.encoder_quantized(x)
        x_recon = self.decode(quantized)

        return loss, x_recon, perplexity

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_vq_conv(z_e)
        return z_e

    def encoder_quantized(self, x):
        z_e = self.encode(x)
        loss, quantized, perplexity, encodings = self.vq_vae(z_e)

        return loss, quantized, perplexity, encodings

    def decode(self, z):
        z = self.post_vq_conv(z)
        y = self.decoder(z)
        return y

    def sample(self, encoding_indices,batch_size,device):

        z_d = self.vq_vae.get_z_d(encoding_indices,(batch_size,self.filter_size,self.filter_size,self.embedding_dim),device)
        y = self.decode(z_d)
        return y

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # shape batch_size,embedding_dim,image_height,image_width
        # 32,32,64,64
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        #print(inputs.size())
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        #print(flat_input.size())
        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        #print(encoding_indices.size())
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

    def get_z_d(self,encoding_indices,z_d_shape,device):
        '''

        :param encoding_indices: sahpe (Batch_size*image_height*image_width,1)
        :param z_d_shape: shape batch_size,image_height,image_width,embedding_dim
        :return:
        '''
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings,device=device)
        encodings.scatter_(1, encoding_indices, 1)
        z_d = torch.matmul(encodings, self._embedding.weight).view(z_d_shape)
        return z_d.permute(0, 3, 1, 2).contiguous()




class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
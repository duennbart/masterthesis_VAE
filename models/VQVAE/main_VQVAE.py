# BSD 3-Clause License
# Copyright (c) 2019, Stefan Dünhuber
# Source Code adapted from https://github.com/zalandoresearch/pytorch-vq-vae.
# Implementing the Vector Quantised Variation AutoEncoder proposed by Aearon et. al in Neural Discrete Representation Learning. https://arxiv.org/abs/1711.00937

from __future__ import print_function
import numpy as np
import json
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import sys

sys.path.insert(0,'/home/stefan/Desktop/Stefan/mastherthesiseval/')
from general.thesislogger import ThesisLogger
from general.utilPytorch.util_pytorch import get_pytorch_dataloaders
from models.VQVAE.VQVAE import VQVAE

from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument("--settings",default='/home/stefan/Desktop/Stefan/mastherthesiseval/models/VQVAE/settings.json', type=str, help="path to settings")

opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(opt.settings) as json_file:
    settings = json.load(json_file)

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
commitment_cost = 0.25
image_size = 256
decay = 0

logger = ThesisLogger(settings=settings,calc_FID_score=True)



train_data_loader,val_data_loader,train_data_loader_4imgs,val_data_loader_4imgs = get_pytorch_dataloaders(train_dataroot=settings["train_dataroot"],val_dataroot=settings["val_dataroot"],
                                                                                                          train_dataroot_4imgs=settings["train_dataroot_4imgs"],val_dataroot_4imgs=settings["val_dataroot_4imgs"],batch_size=settings["batch_size"],img_dim=256)

#data_variance = np.var(training_data.data / 255.0)

model = VQVAE( num_embeddings=settings["num_embeddings"], embedding_dim= settings["embedding_dim"] , commitment_cost=commitment_cost,input_channels = 1,
                 channels=settings["channels"], image_size=image_size, decay=decay).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=settings["learning_rate"], amsgrad=False)

model.train()
train_res_recon_error = []
train_res_perplexity = []

for epoch in range(settings["start_epoch"], settings["epochs"] + 1):

    model.train()

    for iteration, (batch, _) in enumerate(train_data_loader, 0):
        # --------------train------------
        batch = batch.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(batch)



        recon_error = F.mse_loss(data_recon, batch)
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

    print('%d iterations' % (epoch))
    print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
    print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
    print()

    # evaluate model after each epoch
    model.eval()
    with torch.no_grad():
        #generate random sample
        random_indices = torch.mul(torch.rand((100*int(image_size / (2 ** len(settings["channels"])))*int(image_size / (2 ** len(settings["channels"]))),1)),settings["num_embeddings"]).type(torch.cuda.LongTensor)
        fake = model.sample(Variable(random_indices),100,device)
        generated_sample = np.moveaxis(fake.data.cpu().numpy(), 1, -1)
        #print(fake.size)
        # use validation dataset
        iterator = iter(val_data_loader)
        val_inputs, _ = iterator.next()
        val_inputs = val_inputs.to(device)

        _, val_outputs, _ = model(val_inputs)

        val_inputs = val_inputs.data.cpu().numpy()
        val_inputs = np.moveaxis(val_inputs, 1, -1)
        val_outputs = val_outputs.data.cpu().numpy()
        val_outputs = np.moveaxis(val_outputs, 1, -1)

        # use training dataset
        for iteration, (train_inputs, _) in enumerate(train_data_loader, 0):
            train_inputs = train_inputs.to(device)

            _, train_outputs, _ = model(train_inputs)

            if iteration == 0:
                real = train_inputs.data.cpu().numpy()
                rec = train_outputs.data.cpu().numpy()
            else:
                real = np.concatenate((real, train_inputs.data.cpu().numpy()))
                rec = np.concatenate((rec, train_outputs.data.cpu().numpy()))

            if (real.shape[0] >= 100):
                break
        train_inputs = np.moveaxis(real, 1, -1)
        train_outputs = np.moveaxis(rec, 1, -1)

        # use for imgs dataset
        # use validation dataset for imgs
        iterator = iter(val_data_loader_4imgs)
        input_val_4imgs, _ = iterator.next()
        input_val_4imgs = input_val_4imgs.to(device)

        _, output_val_4imgs, _ = model(input_val_4imgs)


        input_val_4imgs = np.moveaxis(input_val_4imgs.data.cpu().numpy(), 1, -1)
        output_val_4imgs = np.moveaxis(output_val_4imgs.data.cpu().numpy(), 1, -1)

        # use train dataset for imgs
        iterator = iter(train_data_loader_4imgs)
        input_train_4imgs, _ = iterator.next()
        input_train_4imgs = input_train_4imgs.to(device)

        _, output_train_4imgs, _ = model(input_train_4imgs)

        input_train_4imgs = np.moveaxis(input_train_4imgs.data.cpu().numpy(), 1, -1)
        output_train_4imgs = np.moveaxis(output_train_4imgs.data.cpu().numpy(), 1, -1)




        lowest_mse_train_path, lowest_mse_test_path = logger.log_epoch(train_input=train_inputs[:100, :],
                                                                       train_recon=train_outputs[:100, :],
                                                                       test_input=val_inputs,
                                                                       test_recon=val_outputs,
                                                                       generated_sample=generated_sample,
                                                                       epoch=epoch, train_4imgs=[input_train_4imgs,
                                                                                                 output_train_4imgs],
                                                                       test_4imgs=[input_val_4imgs, output_val_4imgs])

        # save model for lowest train mse loss
        if (lowest_mse_train_path != None):
            state = {"epoch": epoch, "model": model}
            torch.save(state, lowest_mse_train_path)


        # save model for lowest test mse loss
        if (lowest_mse_test_path != None):
            state = {"epoch": epoch, "model": model}
            torch.save(state, lowest_mse_test_path)

# save after last run
state = {"epoch": epoch, "model": model}
torch.save(state, logger.model_path + 'epoch_' + str(epoch))


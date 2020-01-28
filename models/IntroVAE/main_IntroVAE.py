# BSD 3-Clause License
# Copyright (c) 2019, Stefan Dünhuber
# Source Code adapted from https://github.com/dragen1860/IntroVAE-Pytorch.
# Implementing the IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis proposed by Huang et. al in Neural Discrete Representation Learning. https://arxiv.org/abs/1807.06358
from __future__ import print_function
import argparse
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
import random
import numpy as np
import time
from torchvision.utils import make_grid
import json
import sys
sys.path.insert(0,'/home/stefan/Desktop/Stefan/mastherthesiseval/')
from models.VAE.VAE import VAE
from general.thesislogger import ThesisLogger

from general.utilPytorch.util_pytorch import get_pytorch_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument("--settings",default='/home/stefan/Desktop/Stefan/mastherthesiseval/models/IntroVAE/settings.json', type=str, help="path to settings")

def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)

def record_scalar2(writer, scalar_list_train,scalar_list_val, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list_train):
        writer.add_scalars(scalar_name_list[idx].strip(' '),{scalar_name_list[idx].strip(' ') + '_train': scalar_list_train[idx],scalar_name_list[idx].strip(' ') + '_val': scalar_list_val[idx]}, cur_iter)

def record_image(writer, image_list, cur_iter):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=5), cur_iter)
    
def record_image2(writer, image_list_train,image_list_val, cur_iter):
    image_to_show = torch.cat(image_list_train, dim=0)
    writer.add_image('visualizationTrain', make_grid(image_to_show, nrow=5), cur_iter)

    image_to_show = torch.cat(image_list_val, dim=0)
    writer.add_image('visualizationVal', make_grid(image_to_show, nrow=5), cur_iter)


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)
    # Settings
    # read json settings
    with open(opt.settings) as json_file:
        settings = json.load(json_file)
    nrow = 8
    workers = 12
    cuda = True
    num_vae = settings["epochs_pre_train"]
    logger = ThesisLogger( settings=settings,calc_FID_score=True)

    try:
        os.makedirs(logger.path + '/tensorbaord/')
    except OSError:
        pass

    seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        
    is_scale_back = False
    #--------------build models -------------------------
    model = VAE(cdim=1, hdim=settings["latent_space"], channels=settings['channels'], image_size=settings["input_height"]).cuda()

    if settings["pretrained_model_path"] != False:
        load_model(model, settings["pretrained_model_path"])
    print(model)
            
    optimizerE = optim.Adam(model.encoder.parameters(), lr=settings["learning_rate_encoder"])
    optimizerG = optim.Adam(model.decoder.parameters(), lr=settings["learning_rate_generator"])

    train_data_loader, val_data_loader, train_data_loader_4imgs, val_data_loader_4imgs = get_pytorch_dataloaders(
        train_dataroot=settings["train_dataroot"], val_dataroot=settings["val_dataroot"],
        train_dataroot_4imgs=settings["train_dataroot_4imgs"], val_dataroot_4imgs=settings["val_dataroot_4imgs"],
        batch_size=settings["batch_size"], img_dim=256)


    if settings["tensorboard"]:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=logger.path +'/tensorboard')
    start_time = time.time()
            
    cur_iter = 0
    
    def train_vae(epoch, iteration, batch, cur_iter):
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        real = Variable(batch).cuda()

        loss_info = '[loss_rec, loss_kl]'

        # =========== Update E ================
        real_mu, real_logvar, z, rec = model(real)
        input_img = real.data
        loss_rec = model.reconstruction_loss(rec, real, True)

        loss_kl = model.kl_loss(real_mu, real_logvar).mean()

        loss = loss_rec  + loss_kl

        optimizerG.zero_grad()
        optimizerE.zero_grad()
        loss.backward()
        optimizerE.step()
        optimizerG.step()

        loss_rec = loss_rec.tolist()
        loss_kl = loss_kl.tolist()

        if cur_iter % 1000 is 0:
            if settings["tensorboard"] ==True:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                if cur_iter % 1000 == 0:
                    record_image(writer, [real, rec], cur_iter)

    def train(epoch, iteration, batch, cur_iter):
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        batch_size = batch.size(0)
        
        noise = Variable(torch.zeros(batch_size, settings["latent_space"]).normal_(0, 1)).cuda()
               
        real= Variable(batch).cuda() 


        loss_info = '[loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
            
        #=========== Update E ================ 
        fake = model.sample(noise)            
        real_mu, real_logvar, z, rec = model(real)
        rec_mu, rec_logvar = model.encode(rec.detach())
        fake_mu, fake_logvar = model.encode(fake.detach())
        
        loss_rec =  model.reconstruction_loss(rec, real, True)
        #print(loss_rec)
        lossE_real_kl = model.kl_loss(real_mu, real_logvar).mean()
        lossE_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
        lossE_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()            


        loss_margin = lossE_real_kl + \
                      (F.relu(settings["m_plus"] - lossE_rec_kl) + \
                       F.relu(settings["m_plus"] - lossE_fake_kl))

                    
        lossE = loss_rec  * settings["weight_rec_beta"] + loss_margin * settings["alpha"]

        optimizerE.zero_grad()       
        lossE.backward(retain_graph=True)

        optimizerE.step()
        
        #========= Update G ==================           
        rec_mu, rec_logvar = model.encode(rec)
        fake_mu, fake_logvar = model.encode(fake)
        
        lossG_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
        lossG_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()
        
        lossG = (lossG_rec_kl + lossG_fake_kl)* settings["alpha"] +   loss_rec  * settings["weight_rec_beta"]
                    
        optimizerG.zero_grad()
        lossG.backward()
        # nn.utils.clip_grad_norm(model.decoder.parameters(), 1.0)
        optimizerG.step()

        scalar_list = eval(loss_info)


        
        if cur_iter % 1000 is 0:
            if settings["tensorboard"]:
                 record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                 if cur_iter % 1000 == 0:
                     record_image(writer, [real, rec, fake], cur_iter)

        return scalar_list, loss_info, real, rec, fake


            #----------------Train by epochs--------------------------
    for epoch in range(settings["start_epoch"], settings["epochs"] + 1):

        model.train()
        
        for iteration, (batch,_) in enumerate(train_data_loader, 0):
            #--------------train------------
            if epoch < num_vae:
                train_vae(epoch, iteration, batch, cur_iter)
            else:
                train_scalar_list, loss_info, real, rec, fake = train(epoch, iteration, batch, cur_iter)
            
            cur_iter += 1

        # evaluate model after each epoch
        model.eval()
        with torch.no_grad():
            # generate random sample
            noise = Variable(torch.zeros(100, settings['latent_space']).normal_(0, 1)).cuda()
            fake = model.sample(noise)

            # use validation dataset
            iterator = iter(val_data_loader)
            batch, _ = iterator.next()
            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)

            input_val = Variable(batch).cuda()
            _, _, _, output_val = model(input_val)

            # use training dataset
            train_outputs = []
            for iteration, (batch, _) in enumerate(train_data_loader, 0):
                real1 = Variable(batch).cuda()

                _, _, _, rec1 = model(real1)
                if iteration == 0:
                    real = real1.data.cpu().numpy()
                    rec = rec1.data.cpu().numpy()
                else:
                    real = np.concatenate((real, real1.data.cpu().numpy()))
                    rec = np.concatenate((rec, rec1.data.cpu().numpy()))

                if (real.shape[0] >= 100):
                    break

            # use for imgs dataset
            # use validation dataset for imgs
            iterator = iter(val_data_loader_4imgs)
            batch, _ = iterator.next()
            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)
            input_val_4imgs = Variable(batch).cuda()
            _, _, _, output_val_4imgs = model(input_val_4imgs)
            input_val_4imgs = input_val_4imgs.data.cpu().numpy()
            input_val_4imgs = np.moveaxis(input_val_4imgs, 1, -1)
            output_val_4imgs = output_val_4imgs.data.cpu().numpy()
            output_val_4imgs = np.moveaxis(output_val_4imgs, 1, -1)

            # use train dataset for imgs
            iterator = iter(train_data_loader_4imgs)
            batch, _ = iterator.next()
            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)

            input_train_4imgs = Variable(batch).cuda()
            _, _, _, output_train_4imgs = model(input_train_4imgs)
            input_train_4imgs = input_train_4imgs.data.cpu().numpy()
            input_train_4imgs = np.moveaxis(input_train_4imgs,1, -1)
            output_train_4imgs = output_train_4imgs.data.cpu().numpy()
            output_train_4imgs = np.moveaxis(output_train_4imgs, 1, -1)




            train_inputs = np.moveaxis(real, 1, -1)
            train_outputs = np.moveaxis(rec, 1, -1)
            val_inputs = input_val.data.cpu().numpy()
            val_inputs = np.moveaxis(val_inputs, 1, -1)
            val_outputs = output_val.data.cpu().numpy()
            val_outputs = np.moveaxis(val_outputs, 1, -1)
            generated_sample = np.moveaxis(fake.data.cpu().numpy(), 1, -1)
            lowest_mse_train_path, lowest_mse_test_path = logger.log_epoch(train_input=train_inputs[:100, :],
                                                                           train_recon=train_outputs[:100, :],
                                                                           test_input=val_inputs,
                                                                           test_recon=val_outputs,
                                                                           generated_sample=generated_sample,
                                                                           epoch=epoch,train_4imgs=[input_train_4imgs,output_train_4imgs],test_4imgs=[input_val_4imgs,output_val_4imgs])

            # save model for lowest train mse loss
            if (lowest_mse_train_path != None):
                save_checkpoint(model, lowest_mse_train_path, epoch)

            # save model for lowest test mse loss
            if (lowest_mse_test_path != None):
                save_checkpoint(model, lowest_mse_test_path, epoch)


    #save model
    save_checkpoint(model, logger.model_path + 'weights_epoch_'  + str(epoch) , epoch)

def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()  
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)


def save_checkpoint(model, model_out_path, epoch):
    state = {"epoch": epoch, "model": model}
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))



if __name__ == "__main__":
    main()    
# BSD 3-Clause License
# Copyright (c) 2019, Stefan Dünhuber
# Source Code adapted from https://github.com/divelab/svae.
# Implementing the Spatial Variational Auto-Encoder proposed by Wang et al.:  https://arxiv.org/abs/1705.06821
from __future__ import print_function
import argparse
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torchvision.utils import make_grid
from torch.autograd import Variable
import numpy as np
import json
import random
import sys
import time
sys.path.insert(0,'/home/stefan/Desktop/Stefan/mastherthesiseval/')
from models.SpatialVAE.SpatialVAE import SpatialVAE
from general.thesislogger import ThesisLogger
from general.utilPytorch.util_pytorch import get_pytorch_dataloaders

parser = argparse.ArgumentParser()

parser.add_argument('--nrow', type=int, help='the number of images in each row', default=5)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--cuda', action='store_true', help='enables cuda',default= True)
parser.add_argument('--outf', default='results/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard',default=False, action='store_true', help='enables tensorboard')
parser.add_argument("--pretrained", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--settings",default='/home/stefan/Desktop/Stefan/mastherthesiseval/models/SpatialVAE/settings.json', type=str, help="path to settings")


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
    writer.add_image('visualization', make_grid(image_to_show, nrow=opt.nrow), cur_iter)
    
def record_image2(writer, image_list_train,image_list_val, cur_iter):
    image_to_show = torch.cat(image_list_train, dim=0)
    writer.add_image('visualizationTrain', make_grid(image_to_show, nrow=opt.nrow), cur_iter)

    image_to_show = torch.cat(image_list_val, dim=0)
    writer.add_image('visualizationVal', make_grid(image_to_show, nrow=opt.nrow), cur_iter)


def main():
    
    global opt, model
    opt = parser.parse_args()
    print(opt)

    # read json settings
    with open(opt.settings) as json_file:
        settings = json.load(json_file)

    logger = ThesisLogger(settings=settings,calc_FID_score=True,logger_interval=settings["logger_interval"])

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    is_scale_back = False

    channels = settings['channels']
    #--------------build models -------------------------
    model = SpatialVAE(latent_feature_maps=settings["latent_feature_maps"],latent_feature_size=settings["latent_feature_size"],cdim=1,  channels=channels, image_size=settings['input_height']).cuda()
    print(model)
    if opt.pretrained:
        load_model(model, opt.pretrained)
    #print(model)
            
    optimizerE = optim.Adam(model.encoder.parameters(), lr=settings['learning_rate'])
    optimizerG = optim.Adam(model.decoder.parameters(), lr=settings['learning_rate'])



    train_data_loader, val_data_loader, train_data_loader_4imgs, val_data_loader_4imgs = get_pytorch_dataloaders(
        train_dataroot=settings["train_dataroot"], val_dataroot=settings["val_dataroot"],
        train_dataroot_4imgs=settings["train_dataroot_4imgs"], val_dataroot_4imgs=settings["val_dataroot_4imgs"],
        batch_size=settings["batch_size"], img_dim=256)

    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=opt.outf)
    
    start_time = time.time()
            
    cur_iter = 0
    
    def train_vae(batch, cur_iter):
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        real= Variable(batch).cuda()

        loss_info = '[loss_rec, loss_kl]'
            
        #=========== Update E ================                  
        real_mu, real_logvar, z, rec = model(real) 
        input_img = real.data
        loss_rec =  model.reconstruction_loss(rec, real, True)

        loss_kl = model.kl_loss(real_mu, real_logvar).mean()
                    
        loss = loss_rec+ loss_kl
        
        optimizerG.zero_grad()
        optimizerE.zero_grad()       
        loss.backward()                   
        optimizerE.step() 
        optimizerG.step()

        loss_rec = loss_rec.tolist()
        loss_kl = loss_kl.tolist()

        if cur_iter % 100 is 0:
            info = 'Rec: {:.4f}, KL: {:.4f}, '.format(loss_rec, loss_kl)
            print(info)
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                #if cur_iter % 1000 == 0:
                #    record_image(writer, [real, rec], cur_iter)

            #----------------Train by epochs--------------------------
    for epoch in range(settings["start_epoch"], settings['epochs'] + 1):
        
        model.train()
        
        for iteration, (batch,_) in enumerate(train_data_loader, 0):
            #--------------train------------
            train_vae(batch, cur_iter)
            cur_iter += 1


        # evaluate model after each epoch
        model.eval()

        with torch.no_grad():
            # generate random sample
            noise = Variable(torch.zeros(100,settings['latent_feature_maps'] * settings['latent_feature_size']*settings['latent_feature_size']).normal_(0, 1)).cuda()
            fake = model.sample(noise)

            # use validation dataset
            iterator = iter(val_data_loader)
            batch,_ = iterator.next()
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
                    rec =rec1.data.cpu().numpy()
                else:
                    real = np.concatenate((real,real1.data.cpu().numpy()))
                    rec = np.concatenate((rec, rec1.data.cpu().numpy()))


                train_outputs.append(rec.data)
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
                input_train_4imgs = np.moveaxis(input_train_4imgs, 1, -1)
                output_train_4imgs = output_train_4imgs.data.cpu().numpy()
                output_train_4imgs = np.moveaxis(output_train_4imgs, 1, -1)

            train_inputs = np.moveaxis(real,1,-1)
            train_outputs = np.moveaxis(rec,1,-1)
            val_inputs = input_val.data.cpu().numpy()
            val_inputs = np.moveaxis(val_inputs,1,-1)
            val_outputs = output_val.data.cpu().numpy()
            val_outputs = np.moveaxis(val_outputs,1,-1)
            generated_sample = np.moveaxis(fake.data.cpu().numpy(),1,-1)
            lowest_mse_train_path, lowest_mse_test_path = logger.log_epoch(train_input=train_inputs[:100, :],train_recon=train_outputs[:100,:],
                                                                           test_input=val_inputs,test_recon=val_outputs,
                                                                           generated_sample=generated_sample,epoch=epoch,train_4imgs=[input_train_4imgs,output_train_4imgs],test_4imgs=[input_val_4imgs,output_val_4imgs])


            # save model for lowest train mse loss
            if (lowest_mse_train_path != None):
                 save_checkpoint(model, lowest_mse_train_path,epoch)

             # save model for lowest test mse loss
            if (lowest_mse_test_path != None):
                save_checkpoint(model, lowest_mse_test_path, epoch)


def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()  
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
            
def save_checkpoint(model, model_out_path,epoch):
    state = {"epoch": epoch ,"model": model}
    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()    
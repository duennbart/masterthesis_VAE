# train vae on dsprites to illustrate disentanglement of the latent space

from __future__ import print_function
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,TensorDataset
from torchvision.utils import make_grid
import torch.optim as optim
from torch.autograd import Variable
import torch.nn
import time
import numpy as np

import random
import sys
import tqdm

sys.path.insert(0, '/home/stefan/Desktop/Stefan/mastherthesiseval/')
from models.VAE.VAE import VAE



def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)


def record_scalar2(writer, scalar_list_train, scalar_list_val, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list_train):
        writer.add_scalars(scalar_name_list[idx].strip(' '),
                           {scalar_name_list[idx].strip(' ') + '_train': scalar_list_train[idx],
                            scalar_name_list[idx].strip(' ') + '_val': scalar_list_val[idx]}, cur_iter)


def record_image(writer, image_list, cur_iter):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=5), cur_iter)


def record_image2(writer, image_list_train, image_list_val, cur_iter):
    image_to_show = torch.cat(image_list_train, dim=0)
    writer.add_image('visualizationTrain', make_grid(image_to_show, nrow=5), cur_iter)

    image_to_show = torch.cat(image_list_val, dim=0)
    writer.add_image('visualizationVal', make_grid(image_to_show, nrow=5), cur_iter)


def main():
    channels=[64,64,128]
    latent_space = 10
    input_size = 64
    learning_rate = 0.00001
    batch_size = 256
    epochs = 150
    path = '/home/stefan/Desktop/Stefan/mastherthesiseval/thesis_util/fundamental_plots/dsprites/'
    path_tensorboard = path + 'tensorboard'
    manualSeed = None
    tensorboard = True
    try:
        os.makedirs(path)
    except OSError:
        pass

    if manualSeed is None:
        manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    cudnn.benchmark = True

    # --------------build models -------------------------
    model = VAE(cdim=1, hdim=latent_space, channels=channels, image_size=input_size).cuda()

    # print(model)

    optimizerE = optim.Adam(model.encoder.parameters(), lr=learning_rate)
    optimizerG = optim.Adam(model.decoder.parameters(), lr=learning_rate)

    dsprites = np.load("/home/stefan/Desktop/Stefan/mastherthesiseval/thesis_util/fundamental_plots/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

    imgs = dsprites['imgs']
    np.random.shuffle(imgs)
    imgs= imgs[:60000]
    imgs = np.expand_dims(imgs,1)
    print(imgs.max())
    print(imgs.min())
    print(imgs.shape)
    labels = dsprites['latents_classes']
    labels = labels[:60000]
    tensor_x = torch.stack([torch.Tensor(i) for i in imgs])
    dataset = TensorDataset(tensor_x)  # create your datset
    train_data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)  # create your dataloader


    if tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=path_tensorboard)

    start_time = time.time()

    cur_iter = 0

    def train_vae(batch, cur_iter):
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)

        real = Variable(batch).cuda()

        loss_info = '[loss_rec, loss_kl]'

        # =========== Update E ================
        real_mu, real_logvar, z, rec = model(real)
        input_img = real.data
        loss_rec = model.reconstruction_loss(rec, real, True)

        loss_kl = model.kl_loss(real_mu, real_logvar).mean()

        loss = 50*loss_rec + loss_kl

        optimizerG.zero_grad()
        optimizerE.zero_grad()
        loss.backward()
        optimizerE.step()
        optimizerG.step()


        if cur_iter %1000 is 0:
            if tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                if cur_iter % 1000 == 0:
                    record_image(writer, [real, rec], cur_iter)

                    # ----------------Train by epochs--------------------------

    for epoch in tqdm.tqdm(range(0, epochs+ 1)):

        model.train()

        for iteration, batch in enumerate(train_data_loader, 0):
            # --------------train------------
            train_vae(batch[0], cur_iter)
            cur_iter += 1
            # save after last run
            state = {"epoch": epoch, "model": model}
            torch.save(state, path + 'weights')




    # save model




if __name__ == "__main__":
    main()
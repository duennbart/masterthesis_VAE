from __future__ import print_function
import argparse
from torch.autograd import Variable
import torch.nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
sys.path.insert(0,'/home/stefan/Desktop/Stefan/mastherthesiseval/')
from models.VAE.VAE import VAE
from general.util.ExperimentLoader import ExperimentLoader
from general.utilPytorch.util_pytorch import load_model
from general.utilPytorch.util_pytorch import get_pytorch_dataloaders
from general.util.imagehandler import python_image_grid
plt.gray()
parser = argparse.ArgumentParser()

parser.add_argument("--experiment",default='/home/stefan/Desktop/Stefan/mastherthesiseval/experiments/VAE_07_12PM on November 20, 2019', type=str, help="path to the experiment")

opt = parser.parse_args()

experiment_path = opt.experiment
experimentloader = ExperimentLoader(path2experiment=experiment_path)
settings = experimentloader.getsettings()

# load datasets
_, _, train_data_loader_4imgs, val_data_loader_4imgs = get_pytorch_dataloaders(
    train_dataroot=settings["train_dataroot"], val_dataroot=settings["val_dataroot"],
    train_dataroot_4imgs=settings["train_dataroot_4imgs"], val_dataroot_4imgs=settings["val_dataroot_4imgs"],
    batch_size=settings["batch_size"], img_dim=256)


model = VAE(cdim=1, hdim=settings['latent_space'], channels=settings['channels'], image_size=settings['input_height']).cuda()
load_model(model, experimentloader.get_lowest_mse_train_weights())

# load images as numpy array
iterator = iter(train_data_loader_4imgs)
batch, _ = iterator.next()
input_val_4imgs = Variable(batch).cuda()
input_val_4imgs = input_val_4imgs.data.cpu().numpy()

# selected image
image_idx = 2
input_img = input_val_4imgs[2,:]
input_img = np.expand_dims(input_img,0)


mu, logvar, z, output_img = model(torch.from_numpy(input_img).cuda())
output_img = output_img.data.cpu().numpy()
output_img = output_img.squeeze()
input_img = input_img.squeeze()
print(output_img.shape)

# plot original and reconstrued images
f, ax = plt.subplots(2, figsize=(10, 10))
ax[0].imshow(input_img)
ax[1].imshow(output_img)
ax[0].axis('off')
ax[1].axis('off')


plt.subplots_adjust(left=0.0, bottom=0, right=0.0001, top=0.0001, wspace=0.001, hspace=.001)
plt.tight_layout()
plt.show()

def create_latent_imgs_grid(z_idx_list,number_column,z):
    values_latent_vaiable = np.linspace(-1, 1, number_column)
    recon_imgs = []
    f, ax = plt.subplots(nrows=len(z_idx_list), ncols=number_column, figsize=(30, 30))
    for z_idx in range(0,len(z_idx_list)):
        z_interactive = z.data.cpu().numpy()

        for i in range(0,number_column):

            z_interactive[:, z_idx_list[z_idx]] = values_latent_vaiable[i]
            z_interactive_tensor = torch.from_numpy(z_interactive).cuda()
            recon_latent = model.decode(z_interactive_tensor)
            recon_latent = recon_latent.data.cpu().numpy().reshape(recon_latent.shape[-2], recon_latent.shape[-1])
            recon_imgs.append(recon_latent)
            ax[z_idx][i].imshow(recon_latent)
            ax[z_idx][i].axis('off')


    plt.tight_layout()
    plt.show()
    return recon_imgs
# create latent space plot
recon_imgs = create_latent_imgs_grid(z_idx_list=[1,19,25,38],number_column=5,z=z)

recon_imgs = np.asarray(recon_imgs)
recon_imgs = np.expand_dims(recon_imgs,-1)
recon_imgs = recon_imgs * 255
recon_imgs = recon_imgs.astype(np.uint8)
print(recon_imgs.shape)
grid = python_image_grid(recon_imgs, [4, 5])
grid = np.squeeze(grid)

im = Image.fromarray(grid)

im.save(experiment_path + '/latent_analyze.png')




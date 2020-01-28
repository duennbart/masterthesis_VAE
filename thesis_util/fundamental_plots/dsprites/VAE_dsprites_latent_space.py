from __future__ import print_function
import argparse
from torch.autograd import Variable
import torch.nn
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
sys.path.insert(0,'/home/stefan/Desktop/Stefan/mastherthesiseval/')
from models.VAE.VAE import VAE
from general.utilPytorch.util_pytorch import load_model
from general.util.imagehandler import python_image_grid

plt.gray()
parser = argparse.ArgumentParser()

channels=[64,64,128]
latent_space = 10
input_size = 64
batch_size = 256
epochs = 150
path = '/home/stefan/Desktop/Stefan/mastherthesiseval/thesis_util/fundamental_plots/dsprites/'



# load datasets
dsprites = np.load("/home/stefan/Desktop/Stefan/mastherthesiseval/thesis_util/fundamental_plots/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

imgs = dsprites['imgs']
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

# --------------build models -------------------------
model = VAE(cdim=1, hdim=latent_space, channels=channels, image_size=input_size).cuda()

load_model(model, '/home/stefan/Desktop/Stefan/mastherthesiseval/thesis_util/fundamental_plots/dsprites/weights')

# load images as numpy array
iterator = iter(train_data_loader)
batch = iterator.next()
input_val_4imgs = Variable(batch[0]).cuda()
input_val_4imgs = input_val_4imgs.data.cpu().numpy()

# selected image
image_idx = 10
input_img = input_val_4imgs[image_idx,:]
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

recon_imgs = create_latent_imgs_grid(z_idx_list=list(range(0, latent_space)),number_column=5,z=z)

recon_imgs = np.asarray(recon_imgs)
recon_imgs = np.expand_dims(recon_imgs,-1)
recon_imgs = recon_imgs * 255
recon_imgs = recon_imgs.astype(np.uint8)
print(recon_imgs.shape)
grid = python_image_grid(recon_imgs, [10, 5])
grid = np.squeeze(grid)

im = Image.fromarray(grid)
im.show()
im.save('latent_analyze_dsprites.png')




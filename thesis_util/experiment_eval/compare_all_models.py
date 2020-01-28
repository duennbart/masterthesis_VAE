# create recon and sample images for all models
from thesis_util.thesis_util import stack_trials,create_eval_recon_all_imgs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
from scipy import signal
# for my pc
#path_to_git = r'C:\GIT'
# for tum pc
path_to_git = r'C:\Users\ga45tis\GIT'
save_path = path_to_git + r'\masterthesisgeneral\latex\900 Report\images\experiments\\'


save_path = path_to_git+ r"\masterthesisgeneral\latex\900 Report\images\experiments\\"
title= 'Reconstruction of Test Data'
pdf_file_name='recon_all_experiments_test'
# creat for test images
data = [
    ( path_to_git + r"\mastherthesiseval\experiments\VAE_01_02PM on November 20, 2019\imgs\recon_test_epoch_197.png", r'Input', 0),
    (path_to_git + r"\mastherthesiseval\experiments\VAE_01_02PM on November 20, 2019\imgs\recon_test_epoch_197.png", r'$\textrm{VAE}_{50}$',1),
    (path_to_git + r"\mastherthesiseval\experiments\VAE8192_02_14AM on November 28, 2019\imgs\recon_test_epoch_223.png",r'$\text{VAE}_{8192}$',1),
    (path_to_git + r"\mastherthesiseval\experiments\SpatialVAE_02_14AM on November 21, 2019\imgs\recon_test_epoch_291.png",r'$\text{SVAE}_{3 \times 3 \times 9}$',1),
    (path_to_git + r"\mastherthesiseval\experiments\SpatialVAE161632adpt_05_35PM on November 28, 2019\imgs\recon_test_epoch_292.png",r'$\text{SVAE}_{16 \times 16 \times 32}$',1),
    (path_to_git + r"\mastherthesiseval\experiments\VPGA_04_50AM on November 27, 2019\imgs\recon_test_epoch_247.png",r'$\text{VPGA}_{50}$',1),
    (path_to_git + r"\mastherthesiseval\experiments\VQVAE_10_30AM on November 24, 2019\imgs\recon_test_epoch_243.png",r'$\text{VQ-VAE}_{std}$',1),
    (path_to_git + r"\mastherthesiseval\experiments\VQVAEadapt_06_25AM on November 25, 2019\imgs\recon_test_epoch_292.png",r'$\text{VQ-VAE}_{adpt}$',1),
    (path_to_git + r"\mastherthesiseval\experiments\IntroVAE_02_17AM on November 20, 2019\imgs\recon_test_epoch_300.png",r'$\text{IntroVAE}_{50}$',1)
]
#create_eval_recon_all_imgs(data,title,pdf_file_name,save_directory=save_path,prefix_4include=r"images/experiments/")
# create for training data
data = [
    ( path_to_git + r'\mastherthesiseval\experiments\VAE_01_02PM on November 20, 2019\imgs\recon_train_epoch_290.png', r'Input', 0),
    (path_to_git + r'\mastherthesiseval\experiments\VAE_01_02PM on November 20, 2019\imgs\recon_train_epoch_290.png', r'$\textrm{VAE}_{50}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\VAE8192_02_14AM on November 28, 2019\imgs\recon_train_epoch_293.png',r'$\text{VAE}_{8192}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\SpatialVAE_02_14AM on November 21, 2019\imgs\recon_train_epoch_300.png',r'$\text{SVAE}_{3 \times 3 \times 9}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\SpatialVAE161632adpt_05_35PM on November 28, 2019\imgs\recon_train_epoch_281.png',r'$\text{SVAE}_{16 \times 16 \times 32}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\VPGA_04_50AM on November 27, 2019\imgs\recon_train_epoch_287.png',r'$\text{VPGA}_{50}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\VQVAE_10_30AM on November 24, 2019\imgs\recon_train_epoch_282.png',r'$\text{VQ-VAE}_{std}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\VQVAEadapt_06_25AM on November 25, 2019\imgs\recon_train_epoch_256.png',r'$\text{VQ-VAE}_{adpt}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\IntroVAE_02_17AM on November 20, 2019\imgs\recon_train_epoch_300.png',r'$\text{IntroVAE}_{50}$',1)
]
title= 'Reconstruction of Training Data'
pdf_file_name='recon_all_experiments_train'
#create_eval_recon_all_imgs(data,title,pdf_file_name,save_directory=save_path,prefix_4include=r"images/experiments/")

#create for random generated samples
data = [
    ( path_to_git + r'\mastherthesiseval\experiments\VAE_01_02PM on November 20, 2019\imgs\generated_sample_epoch_300.png', r'Input', 0),
    (path_to_git + r'\mastherthesiseval\experiments\VAE_01_02PM on November 20, 2019\imgs\generated_sample_epoch_300.png', r'$\textrm{VAE}_{50}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\VAE8192_02_14AM on November 28, 2019\imgs\generated_sample_epoch_300.png',r'$\text{VAE}_{8192}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\SpatialVAE_02_14AM on November 21, 2019\imgs\generated_sample_epoch_300.png',r'$\text{SVAE}_{3 \times 3 \times 9}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\SpatialVAE161632adpt_05_35PM on November 28, 2019\imgs\generated_sample_epoch_292.png',r'$\text{SVAE}_{16 \times 16 \times 32}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\VPGA_04_50AM on November 27, 2019\imgs\generated_sample_epoch_300.png',r'$\text{VPGA}_{50}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\VQVAE_10_30AM on November 24, 2019\imgs\generated_sample_epoch_282.png',r'$\text{VQ-VAE}_{std}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\VQVAEadapt_06_25AM on November 25, 2019\imgs\generated_sample_epoch_300.png',r'$\text{VQ-VAE}_{adpt}$',1),
    (path_to_git + r'\mastherthesiseval\experiments\IntroVAE_02_17AM on November 20, 2019\imgs\generated_sample_epoch_299.png',r'$\text{IntroVAE}_{50}$',1)
]
title= 'Random Generated Samples'
pdf_file_name='random_generated_all_experiments'
#create_eval_recon_all_imgs(data,title,pdf_file_name,save_directory=save_path,prefix_4include=r"images/experiments/",add_kl_class=False)
# create learning cuve figures for all experiments
experiments = []

# VAE_50
pathes_2_experiments = [path_to_git + r'\mastherthesiseval\experiments\VAE_01_02PM on November 20, 2019',
                            path_to_git + r'\mastherthesiseval\experiments\VAE_04_12PM on November 20, 2019',
                            path_to_git + r'\mastherthesiseval\experiments\VAE_07_12PM on November 20, 2019',
                            path_to_git + r'\mastherthesiseval\experiments\VAE_11_20PM on November 19, 2019']
model = {"paths": pathes_2_experiments,
         "title":  r'{\normalsize$\textrm{VAE}_{50}$}'}
experiments.append(model)

# VAE_8192
pathes_2_experiments = [path_to_git + r'\mastherthesiseval\experiments\VAE8192_02_14AM on November 28, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\VAE8192_05_56AM on November 28, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\VAE8192_09_41AM on November 28, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\VAE8192_10_33PM on November 27, 2019']

model = {"paths": pathes_2_experiments,
         "title": r'{\normalsize $\textrm{VAE}_{8192}$}'}
experiments.append(model)

# SVAE_339
pathes_2_experiments = [path_to_git + r'\mastherthesiseval\experiments\SpatialVAE_02_14AM on November 21, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\SpatialVAE_05_12AM on November 21, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\SpatialVAE_06_52AM on November 20, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\SpatialVAE_11_16PM on November 20, 2019']
model = {"paths": pathes_2_experiments,
         "title": r'{\normalsize $\textrm{SVAE}_{3 \times 3 \times 9}$}'}
experiments.append(model)

# SVAE_161632
pathes_2_experiments = [path_to_git + r'\mastherthesiseval\experiments\SpatialVAE161632adpt_01_08AM on November 29, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\SpatialVAE161632adpt_01_50PM on November 28, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\SpatialVAE161632adpt_05_35PM on November 28, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\SpatialVAE161632adpt_09_26PM on November 28, 2019']
model = {"paths": pathes_2_experiments,
         "title": r'{\normalsize $\textrm{SVAE}_{16 \times 16 \times 32}$}'}
experiments.append(model)

# VPGA_50
pathes_2_experiments = [path_to_git + r'\mastherthesiseval\experiments\VPGA_01_32PM on November 27, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\VPGA_04_50AM on November 27, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\VPGA_06_51PM on November 26, 2019']
model = {"paths": pathes_2_experiments,
         "title": r'{\normalsize $\textrm{VPGA}_{50}$}'}
experiments.append(model)

# VQ-VAE std
pathes_2_experiments = [path_to_git  +r'\mastherthesiseval\experiments\VQVAE_01_23PM on November 24, 2019',
                        path_to_git  +r'\mastherthesiseval\experiments\VQVAE_04_16PM on November 24, 2019',
                        path_to_git  +r'\mastherthesiseval\experiments\VQVAE_07_09PM on November 24, 2019',
                        path_to_git  +r'\mastherthesiseval\experiments\VQVAE_10_30AM on November 24, 2019']
model = {"paths": pathes_2_experiments,
         "title": r'{\normalsize $\textrm{VQ-VAE}_{std}$}'}
experiments.append(model)

# VQ-VAE adpt
pathes_2_experiments = [path_to_git  +r'\mastherthesiseval\experiments\VQVAEadapt_06_25AM on November 25, 2019',
                        path_to_git  +r'\mastherthesiseval\experiments\VQVAEadapt_07_24PM on November 25, 2019',
                        path_to_git  +r'\mastherthesiseval\experiments\VQVAEadapt_12_05AM on November 25, 2019',
                        path_to_git  +r'\mastherthesiseval\experiments\VQVAEadapt_12_49PM on November 25, 2019']
model = {"paths": pathes_2_experiments,
         "title": r'{\normalsize $\textrm{VQ-VAE}_{adpt}$}'}
experiments.append(model)

# intro vae
pathes_2_experiments = [path_to_git + r'\mastherthesiseval\experiments\IntroVAE_02_17AM on November 20, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\IntroVAE_02_36PM on November 21, 2019',
                        path_to_git + r'\mastherthesiseval\experiments\IntroVAE_09_57AM on November 21, 2019']
model = {"paths": pathes_2_experiments,
         "title":r'{\normalsize $\textrm{IntroVAE}_{50}$}'}
experiments.append(model)

# create MSE plot
xlabel = "Epoch"
ylabel = "MSE"
plot_title = "Average Learning Curve for Test Data "
legend_position='upper right'
epochs = np.arange(0, 300)
fig, ax = plt.subplots()

matplotlib.rcParams['text.usetex'] = True
b, a = signal.butter(1, 0.07)
for element in experiments:
    paths = element["paths"]
    # mse
    result_path_test = [x +  r'\results\mse_test300.npy' for x in paths]
    mse  = stack_trials(result_path_test)
    mse = mse.mean(axis=0)
    mse = mse[:300]
    print(mse.shape)
    title = element["title"]
    mse = signal.filtfilt(b, a, mse)
    ax.plot(epochs, mse, label=title, linewidth=2.5)

#ax.legend(loc=legend_position,ncol=1,bbox_to_anchor=(1.55, 0.8))
ax.grid()
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.set_ylim([0,0.006])
ax.set_xlim([0,300])
ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
plt.xlabel(xlabel)
#plt.ylabel(ylabel)
plt.title(ylabel)

tikzplotlib.save(save_path + "mse_all_test.tex")
plt.show()

# create MSSIM plot
xlabel = "Epoch"
ylabel = "MS-SSIM"
plot_title = "Average Learning Curve for Test Data "
legend_position='upper right'
epochs = np.arange(0, 300)
fig, ax = plt.subplots()

matplotlib.rcParams['text.usetex'] = True
for element in experiments:
    paths = element["paths"]
    # mse
    result_path_test = [x +  r'\results\msssim_test300.npy' for x in paths]
    mse  = stack_trials(result_path_test)
    mse = mse.mean(axis=0)
    mse = mse[:300]
    mse = signal.filtfilt(b, a, mse)
    print(mse.shape)
    title = element["title"]
    ax.plot(epochs, mse, label=title, linewidth=2.5)

#ax.legend(loc=legend_position,ncol=1,bbox_to_anchor=(1.55, 0.8))
ax.grid()
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.set_ylim([0.7,1])
ax.set_xlim([0,300])
ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
plt.xlabel(xlabel)
#plt.ylabel(ylabel)
plt.title(ylabel)

tikzplotlib.save(save_path + "msssim_all_test.tex")
plt.show()


# create FID plot
xlabel = "Epoch"
ylabel = "FID"
plot_title = "Average Learning Curve for Test Data "
legend_position='upper right'
epochs = np.arange(0, 300)
fig, ax = plt.subplots()

matplotlib.rcParams['text.usetex'] = True
for element in experiments:
    paths = element["paths"]
    # mse
    result_path_test = [x +  r'\results\fid_score300.npy' for x in paths]
    mse  = stack_trials(result_path_test)
    mse = mse.mean(axis=0)
    mse = mse[:300]
    mse = signal.filtfilt(b, a, mse)
    print(mse.shape)
    title = element["title"]
    ax.plot(epochs, mse, label=title, linewidth=2.5)

ax.legend(loc=legend_position,ncol=1,bbox_to_anchor=(1.55, 0.8))
ax.grid()
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
#ax.set_ylim([0.7,1])
ax.set_xlim([0,300])
ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
plt.xlabel(xlabel)
#plt.ylabel(ylabel)
plt.title(ylabel)

tikzplotlib.save(save_path + "fid_all_test.tex")
plt.show()

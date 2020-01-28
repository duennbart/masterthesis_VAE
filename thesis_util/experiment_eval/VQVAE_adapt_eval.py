from thesis_util.thesis_util import eval_experiment
from thesis_util.thesis_util import create_eval_recon_imgs,create_eval_random_sample_imgs
# load results for spatial VAE with latent space 3x3x9
# Pathes and names
pathes_2_experiments = [r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VQVAEadapt_06_25AM on November 25, 2019',
                        r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VQVAEadapt_07_24PM on November 25, 2019',
                        r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VQVAEadapt_12_05AM on November 25, 2019',
                        r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VQVAEadapt_12_49PM on November 25, 2019']
save_directory = r'C:\Users\ga45tis\GIT\masterthesisgeneral\latex\900 Report\images\experiments\VQVAE\\'
model_name = 'VQVAE_adpt'
title=r'$\textrm{VQ-VAE}_{adpt}$'
#title='titletest'
sample_img_path =  r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VQVAEadapt_06_25AM on November 25, 2019\imgs\generated_sample_epoch_300.png'
recon_test_img_path =  r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VQVAEadapt_06_25AM on November 25, 2019\imgs\recon_test_epoch_292.png'
recon_train_img_path =  r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VQVAEadapt_06_25AM on November 25, 2019\imgs\recon_train_epoch_256.png'

eval_experiment(save_directory=save_directory,model_name=model_name,pathes_2_experiments=pathes_2_experiments,
                title=title,sample_img_path=sample_img_path,recon_test_img_path=recon_test_img_path,recon_train_img_path=recon_train_img_path)

prefix_4include = r"images/experiments/VQVAE/"
# create for test data
title_test = title+ " - Reconstructions of Test Data"
pdf_file_name = 'recon_test_' + model_name
create_eval_recon_imgs(recon_img_path=recon_test_img_path,title=title_test,pdf_file_name=pdf_file_name,save_directory=save_directory,prefix_4include=prefix_4include)

# create for train data
title_train = title + " - Reconstructions of Training Data"
pdf_file_name = 'recon_train_' + model_name
create_eval_recon_imgs(recon_img_path=recon_train_img_path,title=title_train,pdf_file_name=pdf_file_name,save_directory=save_directory,prefix_4include=prefix_4include)

# create random samples image
title_random_samples = title + " - Random Generated Samples"
pdf_file_name = 'random_generated_' + model_name
create_eval_random_sample_imgs(recon_img_path=sample_img_path, title=title_random_samples, pdf_file_name=pdf_file_name, save_directory=save_directory,prefix_4include=prefix_4include)



from thesis_util.thesis_util import eval_experiment
from thesis_util.thesis_util import create_eval_recon_imgs,create_eval_random_sample_imgs
# load results for VAE with latent space 8192
pathes_2_experiments = [r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VAE8192_02_14AM on November 28, 2019',
                        r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VAE8192_05_56AM on November 28, 2019',
                        r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VAE8192_09_41AM on November 28, 2019',
                        r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VAE8192_10_33PM on November 27, 2019']
save_directory = r'C:\Users\ga45tis\GIT\masterthesisgeneral\latex\900 Report\images\experiments\VAE\\'
model_name = 'VAE_8192'
title=r'$\textrm{VAE}_{8192}$'
#title='titletest'
sample_img_path =  r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VAE8192_02_14AM on November 28, 2019\imgs\generated_sample_epoch_300.png'
recon_test_img_path =  r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VAE8192_02_14AM on November 28, 2019\imgs\recon_test_epoch_223.png'
recon_train_img_path =  r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\VAE8192_02_14AM on November 28, 2019\imgs\recon_train_epoch_293.png'
eval_experiment(save_directory=save_directory,model_name=model_name,pathes_2_experiments=pathes_2_experiments,
                title=title,sample_img_path=sample_img_path,recon_test_img_path=recon_test_img_path,recon_train_img_path=recon_train_img_path)


# create for test data
title_test = title+ " - Reconstructions of Test Data"
pdf_file_name = 'recon_test_' + model_name
create_eval_recon_imgs(recon_img_path=recon_test_img_path,title=title_test,pdf_file_name=pdf_file_name,save_directory=save_directory)

# create for train data
title_train = title + " - Reconstructions of Training Data"
pdf_file_name = 'recon_train_' + model_name
create_eval_recon_imgs(recon_img_path=recon_train_img_path,title=title_train,pdf_file_name=pdf_file_name,save_directory=save_directory)

# create random samples image
title_random_samples = title + " - Random Generated Samples"
pdf_file_name = 'random_generated_' + model_name
create_eval_random_sample_imgs(recon_img_path=sample_img_path, title=title_random_samples, pdf_file_name=pdf_file_name, save_directory=save_directory)
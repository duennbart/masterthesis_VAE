from thesis_util.thesis_util import eval_experiment
from thesis_util.thesis_util import create_eval_recon_imgs,create_eval_random_sample_imgs
# load results for spatial VAE with latent space 3x3x9
# Pathes and names
pathes_2_experiments = [r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\SpatialVAE_02_14AM on November 21, 2019',
                        r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\SpatialVAE_05_12AM on November 21, 2019',
                        r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\SpatialVAE_06_52AM on November 20, 2019',
                        r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\SpatialVAE_11_16PM on November 20, 2019']
save_directory = 'C:\\Users\\ga45tis\\GIT\\masterthesisgeneral\\latex\\900 Report\\images\\experiments\\SpatialVAE\\'
model_name = 'SVAE_339'
title=r'$\textrm{SVAE}_{3 \times 3 \times 9}$'
prefix_4include = r"images/experiments/SpatialVAE/"
sample_img_path =  r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\SpatialVAE_02_14AM on November 21, 2019\imgs\generated_sample_epoch_300.png'
recon_test_img_path =  r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\SpatialVAE_02_14AM on November 21, 2019\imgs\recon_test_epoch_291.png'
recon_train_img_path =  r'C:\Users\ga45tis\GIT\mastherthesiseval\experiments\SpatialVAE_02_14AM on November 21, 2019\imgs\recon_train_epoch_300.png'

eval_experiment(save_directory=save_directory,model_name=model_name,pathes_2_experiments=pathes_2_experiments,
                title=title,sample_img_path=sample_img_path,recon_test_img_path=recon_test_img_path,recon_train_img_path=recon_train_img_path)

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
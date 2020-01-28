from thesis_util.thesis_util import eval_experiment
from thesis_util.thesis_util import create_eval_recon_imgs,create_eval_random_sample_imgs

from PIL import ImageFont, ImageDraw, Image
# load results for VAE with latent space 50

# for my pc
#path_to_git = r'C:\GIT'
# for tum pc




if __name__ == '__main__':
    path_to_git = r'C:\Users\ga45tis\GIT'
    pathes_2_experiments = [path_to_git + r'\mastherthesiseval\experiments\VAE_01_02PM on November 20, 2019',
                            path_to_git + r'\mastherthesiseval\experiments\VAE_04_12PM on November 20, 2019',
                            path_to_git + r'\mastherthesiseval\experiments\VAE_07_12PM on November 20, 2019',
                            path_to_git + r'\mastherthesiseval\experiments\VAE_11_20PM on November 19, 2019']
    save_directory = path_to_git + r'\masterthesisgeneral\latex\900 Report\images\experiments\VAE\\'
    model_name = 'VAE_50'
    title = r'$\textrm{VAE}_{50}$'
    # title='titletest'
    sample_img_path = path_to_git + r'\mastherthesiseval\experiments\VAE_01_02PM on November 20, 2019\imgs\generated_sample_epoch_300.png'
    recon_test_img_path = path_to_git + r'\mastherthesiseval\experiments\VAE_01_02PM on November 20, 2019\imgs\recon_test_epoch_197.png'
    recon_train_img_path = path_to_git + r'\mastherthesiseval\experiments\VAE_01_02PM on November 20, 2019\imgs\recon_train_epoch_290.png'
    eval_experiment(save_directory=save_directory,model_name=model_name,pathes_2_experiments=pathes_2_experiments,
                    title=title,sample_img_path=sample_img_path,recon_test_img_path=recon_test_img_path,recon_train_img_path=recon_train_img_path)

    prefix_4include = r"images/experiments/VAE/"
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
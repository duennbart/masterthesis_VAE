from general.util.fid_official_tf import *

path = '/home/biomech/Documents/OsteoData/KneeXrayData/ClsKLData/kneeKL299_2/train'
inception_path = '/home/stefan/Desktop/Stefan/mastherthesiseval/general/inception_model'
save_path = '/home/stefan/Desktop/Stefan/mastherthesiseval/general/pre_calc_fid/mu_sigma_train.npz'
#calcuate mu and sigma for trainings data
mu,sigma = save_mu_sigma_given_paths(path,save_path, inception_path, low_profile=False)
print(mu.shape)

paths = []
paths.append('/home/stefan/Desktop/Stefan/mastherthesiseval/general/pre_calc_fid/mu_sigma_train.npz')
paths.append('/home/biomech/Documents/OsteoData/KneeXrayData/ClsKLData/kneeKL299_2/val')
# compare fid score of trainings data with validation data
fid_value = calculate_fid_given_paths(paths, inception_path, low_profile=False)
print("FID: ", fid_value)
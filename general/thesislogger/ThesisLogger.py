import numpy as np
import datetime
import os
import glob
import json
import matplotlib
import matplotlib.pyplot as plt

#own imports
from general.util.mssim import MultiScaleSSIM
from general.util.imagehandler import save_reconimgs_as_grid
from general.util.fid_official_tf import calculate_fid_given_array

matplotlib.rcParams['text.usetex'] = True
def MSE(input,recon):
    return (np.square(input- recon)).mean(axis=None)



class ThesisLogger():

    def __init__(self,settings,calc_FID_score=False,logger_interval=1):

        modelname = settings['model_name'].replace('{','').replace('}','')
        self.pre_calc_mu_sigma = '/home/stefan/Desktop/Stefan/mastherthesiseval/general/pre_calc_fid/mu_sigma_train.npz'
        self.inception_path = '/home/stefan/Desktop/Stefan/mastherthesiseval/general/inception_model'
        self.path = settings['thesis_logger_path'] + '/' + modelname + '_' + datetime.datetime.now().strftime("%I_%M%p on %B %d, %Y")
        self.calc_FID_score = calc_FID_score
        self.logger_interval = logger_interval
        #create init path
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        #else:
        #    raise NameError('Directory for logger already exists')

        # save settings
        self.settings = settings
        with open( self.path + '/settings.json', 'w') as outfile:
            json.dump(settings, outfile)


        # create path to save the model weights
        self.model_path = self.path +'/model_weights/'
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        # create path where to save the results
        self.result_path = self.path +'/results/'
        if not os.path.isdir(self.result_path):
            os.makedirs(self.result_path)

        # create path where to save the images
        self.imgs_path = self.path + '/imgs/'
        if not os.path.isdir(self.imgs_path):
            os.makedirs(self.imgs_path)

        self.history_fid = []
        self.history_epoch = []
        self.history_mse_train = []
        self.history_mse_test = []
        self.history_msssim_train = []
        self.history_msssim_test = []
        self.history_epoch_fid = []
        self.lowest_mse_train = 100000
        self.lowest_mse_test =  100000


    def log_epoch(self,train_input,train_recon,test_input,test_recon,generated_sample,epoch,train_4imgs,test_4imgs):
        '''
        logs the mse, ms-ssim every epoch and the fid score every 10 epochs
        Creates a figrure for mse, ms-ssim and fid
        saves the numpy history array of mse, ms-ssim and fid
        creates 5x4 images from the train_imgs_plot and test_imgs_plot

        :param train_input: 100 random samples from the training dataset (N,H,W,C)
        :param train_recon: the rconstructed train_input samples (N,H,W,C)
        :param test_input: 100 random samples from the test dataset (N,H,W,C)
        :param test_recon: the rconstructed test_input samples (N,H,W,C)
        :param generated_sample: 10 generated samples (N,H,W,C)
        :param epoch: number of the current epoch
        :param train_4imgs: always the same 10 input images from the training dataset, a figure will be created out of it, List with input and output images
        :return:test_4imgs: always the same 10 input images from the test dataset, a figure will be created out of it, List with input and output images
        '''
        # check interval
        if epoch % self.logger_interval != 0:
            return None,None

        # check dimensions
        assert train_input.ndim == 4
        assert test_input.ndim == 4
        assert train_input.shape == train_recon.shape
        assert test_input.shape == test_recon.shape
        # check value range
        assert (train_input.max() <= 1) and (train_input.min() >= 0)
        assert (train_recon.max() <= 1) and (train_recon.min() >= 0)
        assert (test_input.max() <= 1) and (test_input.min() >= 0)
        assert (test_recon.max() <= 1) and (test_recon.min() >= 0)

        self.history_epoch.append(epoch)
        # calculate mse error
        self.history_mse_train.append(MSE(train_input,train_recon))
        self.history_mse_test.append(MSE(test_input,test_recon))

        # calculate msssim
        self.history_msssim_train.append(MultiScaleSSIM(train_input,train_recon,max_val=1))
        self.history_msssim_test.append(MultiScaleSSIM(test_input,test_recon,max_val=1))

        if self.calc_FID_score == True and epoch % 1 == 0:
            # sample must be between 0-255
            samples  = (generated_sample*255).astype(np.uint8)
            samples = samples.tolist()
            imgs = []
            for sample in samples:
                sample = np.asarray(sample)
                sample = np.squeeze(sample,2)
                stacked_img = np.stack((sample,) * 3, axis=-1)


                imgs.append(stacked_img)


            color_img = np.asarray(imgs)
            fid = calculate_fid_given_array(pre_calc_path=self.pre_calc_mu_sigma, x=color_img, inception_path=self.inception_path, low_profile=False)
            self.history_fid.append(fid)
            self.history_epoch_fid.append(epoch)




        if epoch % 10== 0:
            # save history as numpy array
            self.save_history(epoch)
            # create figures
            self.create_mse_figure()
            self.create_msssim_figure()
        if self.calc_FID_score == True  and epoch % 10 == 0:
            self.create_fid_figure()

        #check lowest mse
        lowest_mse_train_path = None
        if (self.history_mse_train[-1] < self.lowest_mse_train):
            self.lowest_mse_train = self.history_mse_train[-1]
            #lowest mse for train dataset
            lowest_mse_train_path =  self.model_path + 'lowest_mse_train_epoch_' + str(epoch)
            # delete old  files
            oldFiles = glob.glob(self.model_path + "lowest_mse_train_epoch_*")
            for deletefile in oldFiles:
                os.remove(deletefile)

        lowest_mse_test_path = None
        if (self.history_mse_test[-1] < self.lowest_mse_test):
            self.lowest_mse_test = self.history_mse_test[-1]
            #lowest mse for train dataset
            lowest_mse_test_path = self.model_path + 'lowest_mse_test_epoch_' + str(epoch)
            # delete old  files
            oldFiles = glob.glob(self.model_path + "lowest_mse_test_epoch_*")
            for deletefile in oldFiles:
                os.remove(deletefile)

        # save the images

        save_reconimgs_as_grid(input_imgs=train_4imgs[0],output_imgs=train_4imgs[1], path= self.imgs_path + 'recon_train_epoch_' + str(epoch)+ '.png')
        save_reconimgs_as_grid(input_imgs=test_4imgs[0], output_imgs=test_4imgs[1],path = self.imgs_path + 'recon_test_epoch_' + str(epoch)+ '.png')

        if isinstance(generated_sample, np.ndarray):
            # save generated samples
            save_reconimgs_as_grid(input_imgs=train_input, output_imgs=generated_sample,
                                   path=self.imgs_path + 'generated_sample_epoch_' + str(epoch) + '.png')
        print("Epoch: " + str(epoch) + "MSE train: " + str(self.history_mse_train[-1]))


        return lowest_mse_train_path, lowest_mse_test_path

    def create_mse_figure(self):
        fig, ax = plt.subplots()

        ax.plot(self.history_epoch, self.history_mse_train, 'b', label='Train Data',)
        ax.plot(self.history_epoch, self.history_mse_test, 'r', label='Validation Data')
        legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
        out_path = self.result_path + 'mse_history.svg'
        plt.xlabel(r'Epoch')
        plt.ylabel(r'MSE')
        plt.title(r'$\textrm{' + self.settings["model_name"] + '}$')
        plt.savefig(out_path)
        plt.close()

    def create_msssim_figure(self):
        fig, ax = plt.subplots()
        ax.plot(self.history_epoch, self.history_msssim_train, 'b', label='Train Data')
        ax.plot(self.history_epoch, self.history_msssim_test, 'r', label='Validation Data')
        legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
        out_path = self.result_path + 'msssim_history.svg'
        plt.xlabel(r'Epoch')
        plt.ylabel(r'MS-SSIM')
        plt.title(r'$\textrm{' + self.settings["model_name"] + '}$')
        plt.savefig(out_path)
        plt.close()

    def create_fid_figure(self):
        fig, ax = plt.subplots()
        ax.plot(self.history_epoch_fid, self.history_fid, 'b', label='FID score')
        legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
        out_path = self.result_path + 'fid_history.svg'
        plt.xlabel(r'Epoch')
        plt.ylabel(r'FID')
        plt.title(r'$\textrm{' + self.settings["model_name"] + '}$')
        plt.savefig(out_path)
        plt.close()


    def save_history(self,epoch):
        # delete old .npy files
        oldFiles = glob.glob(self.result_path +"*.npy")
        for deletefile in oldFiles:
            os.remove(deletefile)
        #save mse
        np.save('{}mse_train{}.npy'.format(self.result_path, epoch), np.asarray(self.history_mse_train))
        np.save('{}mse_test{}.npy'.format(self.result_path, epoch), np.asarray(self.history_mse_test))

        #save msssim
        np.save('{}msssim_train{}.npy'.format(self.result_path, epoch), np.asarray(self.history_msssim_train))
        np.save('{}msssim_test{}.npy'.format(self.result_path, epoch), np.asarray(self.history_msssim_test))

        #save fid score
        if self.calc_FID_score == True  and epoch % 10 == 0:
            np.save('{}fid_score{}.npy'.format(self.result_path, epoch), np.asarray(self.history_fid))







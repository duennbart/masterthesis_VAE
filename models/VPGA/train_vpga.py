# BSD 3-Clause License
# Copyright (c) 2019, Stefan Dünhuber
# Source Code adapted from https://github.com/zj10/PGA.
# Implementing the Perceptual Generative Autoencoders proposed by Zhang et al.:  https://https://arxiv.org/abs/1906.10335

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
from functools import partial
import json
import traceback

#import imlib as im
import numpy as np
#import pylib
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import sys
sys.path.insert(0,'/home/stefan/Desktop/Stefan/mastherthesiseval/')
from general.utilTF1.dataset import Dataset,get_dataset
from general.utilTF1.models import conv_knee_model
from general.utilTF1.utils import summary,session,load_checkpoint
from general.thesislogger import ThesisLogger

# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--settings",default='/home/stefan/Desktop/Stefan/mastherthesiseval/models/VPGA/settings.json', type=str, help="path to settings")

args = parser.parse_args()

# Settings
# read json settings
with open(args.settings) as json_file:
    settings = json.load(json_file)

epoch = settings["epochs"]
batch_size = settings["batch_size"]
lr = settings["learning_rate"]
use_bn = True
z_dim = settings["latent_space"]
zn_rec_coeff = settings["zn_rec_coeff"]
zh_rec_coeff = settings["zh_rec_coeff"]
vrec_coeff = settings["vrec_coeff"]
vkld_coeff = settings["vkld_coeff"]

# init logger
logger = ThesisLogger( settings=settings,
                          calc_FID_score=True)




# get datasets
#train data set
Dataset, img_shape, get_imgs = get_dataset(settings["train_dataroot"])
dataset_train = Dataset(batch_size=batch_size)

# validation dataset
Dataset_val, img_shape, get_imgs = get_dataset(settings["val_dataroot"])
dataset_val = Dataset_val(batch_size=100)

#data sets for imgs
Dataset_train_4imgs, _, _ = get_dataset(settings["train_dataroot_4imgs"],shuffle=False)
dataset_train_4imgs = Dataset_train_4imgs(batch_size=10)

Dataset_val_4imgs, _, _ = get_dataset(settings["val_dataroot_4imgs"],shuffle=False)
dataset_val_4imgs = Dataset_val_4imgs(batch_size=10)

Enc, Dec =  conv_knee_model()
Enc = partial(Enc, z_dim=z_dim, use_bn=use_bn, sigma=True)
Dec = partial(Dec, channels=img_shape[2], use_bn=use_bn)


# ==============================================================================
# =                                    graph                                   =
# ==============================================================================

def enc_dec(img, is_training=True):
    # encode
    z_mu, z_log_sigma_sq = Enc(img, is_training=is_training)

    # decode
    img_rec = dec(z_mu, is_training=is_training)

    return z_mu, z_log_sigma_sq, img_rec

def dec(z_mu,is_training=True):
    img_rec = Dec(z_mu,is_training=is_training)
    return img_rec


def dec_enc(z, is_training=True, no_enc_grad=False):
    # decode
    img = Dec(z, is_training=is_training)

    # encode
    z_rec, _ = Enc(img, is_training=is_training)
    if no_enc_grad:
        z_rec -= Enc(tf.stop_gradient(img), is_training=is_training)[0] - tf.stop_gradient(z_rec)

    return z_rec


# input
img = tf.placeholder(tf.float32, [None] + img_shape)
normal_dist = tfd.MultivariateNormalDiag(scale_diag=np.ones([z_dim], dtype=np.float32))

# encode & decode
z_mu, z_log_sigma_sq, img_rec = enc_dec(img)
z_noise = tf.exp(0.5 * z_log_sigma_sq) * tf.random_normal(tf.shape(z_mu))
zn_targ, zh_targ = normal_dist.sample(batch_size), tf.stop_gradient(z_mu)
zn_rec, zh_rec = dec_enc(zn_targ), dec_enc(zh_targ)
z_mu_rec, z_rec = dec_enc(tf.stop_gradient(z_mu), no_enc_grad=True), \
                  dec_enc(tf.stop_gradient(z_mu) + z_noise, no_enc_grad=True)

z_placeholder = tf.placeholder(tf.float32,[None] + [z_dim])
# encode & decode
img_rec_from_z = dec(z_placeholder)

# loss
img_rec_loss = tf.losses.mean_squared_error(img, img_rec)
zn_rec_loss, zh_rec_loss = tf.losses.mean_squared_error(zn_targ, zn_rec), tf.losses.mean_squared_error(zh_targ, zh_rec)
z_mu_norm = tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(z_mu), 0)))
vrec_loss = tf.losses.mean_squared_error(z_mu_rec / z_mu_norm, z_rec / z_mu_norm)
vkld_loss = -tf.reduce_mean(0.5 * (1 + z_log_sigma_sq - z_mu ** 2 - tf.exp(z_log_sigma_sq)))


enc_loss = img_rec_loss + zn_rec_coeff * zn_rec_loss + vrec_coeff * vrec_loss + vkld_coeff * vkld_loss




if zh_rec_coeff > 0:
    enc_loss += zh_rec_coeff * zh_rec_loss
dec_loss = img_rec_loss + vrec_coeff * vrec_loss

# otpim
enc_vars = []
dec_vars = []
for var in tf.trainable_variables():
    if var.name.startswith('Enc'):
        enc_vars.append(var)
    elif var.name.startswith('Dec'):
        dec_vars.append(var)
optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
enc_gvs = optimizer.compute_gradients(enc_loss, enc_vars)
dec_gvs = optimizer.compute_gradients(dec_loss, dec_vars)
train_op = optimizer.apply_gradients(enc_gvs + dec_gvs)

# summary
summary = summary({img_rec_loss: 'img_rec_loss',
                      zn_rec_loss: 'zn_rec_loss', zh_rec_loss: 'zh_rec_loss',
                      vrec_loss: 'vrec_loss', vkld_loss: 'vkld_loss'})


# sample

z_intp_sample, z_log_sigma_sq_sample, img_rec_sample = enc_dec(img, is_training=True)

fid_sample = Dec(normal_dist.sample([100]), is_training=True)


z_intp_split, img_split = tf.split(z_intp_sample, 2), tf.split(img, 2)
img_intp_sample = [Dec((1 - i) * z_intp_split[0] + i * z_intp_split[1], is_training=True) for i in np.linspace(0, 1, 9)]
img_intp_sample = [img_split[0]] + img_intp_sample + [img_split[1]]
img_intp_sample = tf.concat(img_intp_sample, 2)


# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# session
sess = session()

# saver
saver = tf.compat.v1.train.Saver()

# summary writer
summary_writer = tf.compat.v1.summary.FileWriter(logger.path + '/tensorboard', sess.graph)

if settings["pretrained_model_path"] != False :
    try:
        load_checkpoint(settings["pretrained_model_path"], sess)
    except:
        sess.run(tf.global_variables_initializer())
else:
    sess.run(tf.global_variables_initializer())


# train
try:
    img_ipt_sample,label = get_imgs(dataset_val.get_next())
    z_ipt_sample = np.random.normal(size=[100, z_dim])

    losses = {img_rec_loss: 'img_rec_loss',
              zn_rec_loss: 'zn_rec_loss', zh_rec_loss: 'zh_rec_loss',
              vrec_loss: 'vrec_loss', vkld_loss: 'vkld_loss'}
    it = -1

    for ep in range(epoch):
        dataset_train.reset()

        it_per_epoch = it_in_epoch if it != -1 else -1
        it_in_epoch = 0

        for batch in dataset_train:
            it += 1
            it_in_epoch += 1

            # batch data
            img_ipt,label = get_imgs(batch)

            sess.run([train_op], feed_dict={img: img_ipt})


        # validate model
        #training data
        dataset_train.reset()
        for iteration,batch in enumerate(dataset_train):
            # batch data
            img_ipt,label = get_imgs(batch)

            summary_opt, img_rec_train = sess.run([summary,  img_rec_sample],
                                                                  feed_dict={img: img_ipt})
            if iteration == 0:
                train_input = img_ipt
                train_output = img_rec_train
            else:
                train_input = np.concatenate((train_input, img_ipt))
                train_output = np.concatenate((train_output, img_rec_train))

            if (train_input.shape[0] >= 100):
                break

        summary_writer.add_summary(summary_opt, ep)

        #test data
        dataset_val.reset()
        iterator = iter(dataset_val)
        batch = iterator.next()
        img_ipt_val, label = get_imgs(batch)

        img_rec_val = sess.run([img_rec_sample],feed_dict={img: img_ipt_val})

        # train data for images
        dataset_train_4imgs.reset()
        iterator = iter(dataset_train_4imgs)
        batch = iterator.next()
        img_ipt_train_4imgs, label = get_imgs(batch)

        img_rec_train_4imgs = sess.run([img_rec_sample], feed_dict={img: img_ipt_train_4imgs})

        # valdata for images
        dataset_val_4imgs.reset()
        iterator = iter(dataset_val_4imgs)
        batch = iterator.next()
        img_ipt_val_4imgs, label = get_imgs(batch)

        img_rec_val_4imgs = sess.run([img_rec_sample], feed_dict={img: img_ipt_val_4imgs})

        #generate random sample
        generated_random_sample = sess.run(fid_sample)

        lowest_mse_train_path, lowest_mse_test_path = logger.log_epoch(train_input=train_input[:100, :],
                                                                       train_recon=train_output[:100, :],
                                                                       test_input=np.asarray(img_ipt_val),
                                                                       test_recon=np.asarray(img_rec_val).squeeze(0),
                                                                       generated_sample=generated_random_sample,
                                                                       epoch=ep, train_4imgs=[img_ipt_train_4imgs,
                                                                                                 img_rec_train_4imgs[0]],
                                                                       test_4imgs=[img_ipt_val_4imgs, img_rec_val_4imgs[0]])



        # save model for lowest train reconstruction mse
        if (lowest_mse_train_path != None):

            save_path = saver.save(sess, lowest_mse_train_path + '.ckpt')
            print('Model is saved in file: %s' % save_path)

        # save model for lowest test reconstruction mse
        if (lowest_mse_test_path != None):
            save_path = saver.save(sess, lowest_mse_test_path + '.ckpt')
            print('Model is saved in file: %s' % save_path)

except:
    traceback.print_exc()
finally:
    sess.close()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from general.utilTF1.ops.layers import flatten_fully_connected

conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)


def conv_knee_model():
    def Enc(img, z_dim, dim=8, use_bn=False, is_training=True, sigma=False):
        bn = partial(batch_norm, is_training=is_training) if use_bn else None
        conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)
        conv_lrelu = partial(conv, normalizer_fn=None, activation_fn=lrelu)

        with tf.variable_scope('Enc', reuse=tf.AUTO_REUSE):
            y = conv_bn_lrelu(img, dim, 4, 2)#8
            y = residual_blockDown(y, dim*2,is_training)#16
            y = residual_blockDown(y, dim * 4, is_training)#32
            y = residual_blockDown(y, dim * 8, is_training)#64
            y = residual_blockDown(y, dim * 16, is_training)#128
            y = residual_blockDown(y, dim * 32, is_training)#256

            z_mu = fc(y, z_dim)
            if sigma:
                z_log_sigma_sq = fc(y, z_dim, biases_initializer=tf.constant_initializer(2. * np.log(0.1)))
                return z_mu, z_log_sigma_sq
            else:
                return z_mu

    def Dec(z, dim=8, channels=1, use_bn=False, is_training=True):
        bn = partial(batch_norm, is_training=is_training) if use_bn else None
        dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)
        dconv_relu = partial(dconv, normalizer_fn=None, activation_fn=relu)

        with tf.variable_scope('Dec', reuse=tf.AUTO_REUSE):
            y = relu(fc(z, 4 * 4 * dim * 32))
            y = tf.reshape(y, [-1, 4, 4, dim * 32])
            y = residual_blockUp(y, dim * 16, is_training) #8x8x128
            y = residual_blockUp(y, dim * 8, is_training)  # 16x16x64
            y = residual_blockUp(y, dim * 4, is_training)  # 32x32x32
            y = residual_blockUp(y, dim * 2, is_training)  # 64x64x16
            y = residual_blockUp(y, dim , is_training)  # 128x128x8

            y = dconv_bn_relu(y, dim , 4, 2)#256x256x4

            img = tf.sigmoid(dconv(y, channels, 4, 1))
            return img

    return Enc, Dec


def residual_blockDown(input_layer, output_channel,is_training,block_number=1, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    bn = partial(batch_norm, is_training=is_training)
    conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu)

    input_channel = input_layer.get_shape().as_list()[-1]


    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    conv1 = conv_bn_lrelu(inputs=input_layer, num_outputs=output_channel, kernel_size=4, stride=1, padding='SAME')


    conv2 = conv_bn_lrelu(inputs=conv1, num_outputs=output_channel, kernel_size=4, stride=2, padding='SAME')

    padded_input = conv_bn_lrelu(inputs=input_layer, num_outputs=output_channel, kernel_size=1, stride=2, padding='SAME')

    output = conv2 + padded_input
    return output

def residual_blockUp(input_layer, output_channel,is_training,block_number=1, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    bn = partial(batch_norm, is_training=is_training)
    dconv_bn_relu = partial(dconv, normalizer_fn=bn, activation_fn=relu)

    input_channel = input_layer.get_shape().as_list()[-1]


    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    conv1 = dconv_bn_relu(inputs=input_layer, num_outputs=output_channel, kernel_size=4, stride=1, padding='SAME')


    conv2 = dconv_bn_relu(inputs=conv1, num_outputs=output_channel, kernel_size=4, stride=2, padding='SAME')

    padded_input = dconv_bn_relu(inputs=input_layer, num_outputs=output_channel, kernel_size=1, stride=2, padding='SAME')

    output = conv2 + padded_input
    return output
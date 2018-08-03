###########################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
###########################################

import numpy as np
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average


def conv_bn_relu(x, filter_shape, stride):

    w = create_variable(name='conv', shape=filter_shape)

    # convolution, no need for bias due to BN'mean calc
    conv_out = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

    bn_out = batch_norm(conv_out, train=True)

    activation = tf.nn.relu(bn_out)

    return activation


def bn_relu_conv(x, filter_shape, stride):

    bn_out = batch_norm(x, train=True)

    activation = tf.nn.relu(bn_out)

    w = create_variable(name='conv', shape=filter_shape)

    # convolution, no need for bias due to BN'mean calc
    conv_out = tf.nn.conv2d(activation, w, strides=[1, stride, stride, 1], padding='SAME')

    return conv_out


def conv2d(x, filter_shape, stride, padding, name='conv', verbose=True):

    w = create_variable(name + '_w', shape=filter_shape)

    b = create_variable(name + '_b', shape=[filter_shape[-1]], initializer=tf.constant_initializer(0.05))

    xw = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)

    xw_b = tf.nn.bias_add(xw, b)

    if verbose:
        print("> %s layer: knum=%d, ksize=%d, stride=%d, output_shape=%s"
              % (name, filter_shape[-1], filter_shape[0], stride, str(xw_b.get_shape())))

    return xw_b


def relu(x):

    activation = tf.nn.relu(x)

    return activation


def max_pool(x, ksize, stride, padding):

    out = tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    return out


def dense(x, num_outputs, use_relu, name='fc', verbose=True):

    num_inputs = x.get_shape().as_list()[-1]

    w = create_variable(name + '_w', shape=[num_inputs, num_outputs])

    b = create_variable(name + '_b', shape=[num_outputs], initializer=tf.constant_initializer(0.05))

    act = tf.matmul(x, w) + b

    if use_relu:
        act = tf.nn.relu(act)

    if verbose:
        print("> %s layer: num_in=%d, num_out=%d, output_shape=%s"
              % (name, num_inputs, num_outputs, str(act.get_shape())))

    return act


def flatten(x):
    nums = np.prod(x.get_shape().as_list()[1:])  # C*W*H
    return tf.reshape(x, [-1, nums])  # samples * (C*W*H)


def batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name='bn'):

    depth = tf.shape(x)[-1]

    moving_mean = create_variable(name + '_mu', shape=[depth], initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = create_variable(name + '_sigma', shape=[depth], initializer=tf.ones_initializer(), trainable=False)

    def mean_var_with_update():
        mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
        with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                      assign_moving_average(moving_variance, variance, decay)]):
            return tf.identity(mean), tf.identity(variance)

    mean, variance = tf.cond(train, mean_var_with_update(), lambda: (moving_mean, moving_variance))

    if affine:
        beta = tf.get_variable(name + '_beta', shape=[depth], initializer=tf.zeros_initializer())
        gamma = tf.get_variable(name + '_gamma', shape=[depth], initializer=tf.ones_initializer())

        x = tf.nn.batch_norm_with_global_normalization(x, mean, variance, beta, gamma, eps)
    else:
        x = tf.nn.batch_norm_with_global_normalization(x, mean, variance, None, None, eps)

    return x


def lrn(x, radius, alpha, beta, bias=1.0):

    out = tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

    return out


def create_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True):

    # regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)

    new_var = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=None, trainable=trainable)

    return new_var


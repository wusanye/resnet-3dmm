###########################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
###########################################

import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average


def conv2d(x, in_planes, planes, ksize, stride, padding, name='conv', verbose=True):

    with tf.name_scope(name):

        w = tf.Variable(tf.truncated_normal([ksize, ksize, in_planes, planes], stddev=0.05), name='w')

        b = tf.Variable(tf.constant(0.05, shape=[planes]), name='b')

        xw = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)

        xw_b = tf.nn.bias_add(xw, b)

    if verbose:
        print("> %s layer: knum=%d, ksize=%d, stride=%d, output_shape=%s"
              % (name, planes, ksize, stride, str(xw_b.get_shape())))

    return xw_b


def relu(x):

    activation = tf.nn.relu(x)

    return activation


def max_pool(x, ksize, stride, padding):

    out = tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    return out


def dense(x, num_outputs, use_relu, name='fc', verbose=True):

    num_inputs = x.get_shape().as_list()[-1]

    with tf.name_scope(name):

        w = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05), name='w')

        b = tf.Variable(tf.constant(0.05, shape=[num_outputs]), name='b')

        act = tf.matmul(x, w) + b

        if use_relu:
            act = tf.nn.relu(act)

        if verbose:
            print("> %s layer: num_in=%d, num_out=%d, output_shape=%s"
                  % (name, num_inputs, num_outputs, str(act.get_shape())))

    return act


def batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name='bn'):

    depth = tf.shape(x)[-1]

    with tf.name_scope(name):
        moving_mean = tf.Variable(tf.zeros([depth]), name='mean', trainable=False)
        moving_variance = tf.Variable(tf.ones([depth]), name='variance', trainable=False)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)

        mean, variance = tf.cond(train, mean_var_with_update(), lambda: (moving_mean, moving_variance))

        if affine:
            beta = tf.Variable(tf.zeros([depth]), name='beta')
            gamma = tf.Variable(tf.ones([depth]), name='gamma')

            x = tf.nn.batch_norm_with_global_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_norm_with_global_normalization(x, mean, variance, None, None, eps)

        return x


def lrn(x, radius, alpha, beta, bias=1.0):

    out = tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

    return out

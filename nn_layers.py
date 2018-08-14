###########################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
###########################################

import numpy as np
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average


def residual_block(block_func, x, out_depth, stride, first_resblk=False, training=False):

    return block_func(x, out_depth, stride, first_resblk, training)


'''ResNet V2 with Pre-Activation'''


def basic_block(x, out_depth, stride, first_resblk=False, training=False):

    input_depth = x.get_shape().as_list()[-1]

    with tf.variable_scope('convA'):

        if first_resblk:
            w = create_variable(name='conv_w', shape=[3, 3, input_depth, out_depth])
            conv1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv(x, [3, 3, input_depth, out_depth], stride, training)

    with tf.variable_scope("convB"):

        conv2 = bn_relu_conv(conv1, [3, 3, out_depth, out_depth], 1, training)

    block_out = shortcut(x, conv2)

    return block_out


def bottleneck(x, out_depth, stride, first_resblk=False, training=False):

    input_depth = x.get_shape().as_list()[-1]

    with tf.variable_scope('convA'):

        if first_resblk:
            w = create_variable(name='conv_w', shape=[1, 1, input_depth, out_depth])
            conv1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv(x, [1, 1, input_depth, out_depth], stride, training)

    with tf.variable_scope("convB"):

        conv2 = bn_relu_conv(conv1, [3, 3, out_depth, out_depth], 1, training)

    with tf.variable_scope("convC"):

        conv3 = bn_relu_conv(conv2, [1, 1, out_depth*4, out_depth], 1, training)

    conv_block_out = shortcut(x, conv3)

    return conv_block_out


def conv_bn_relu(x, filter_shape, stride, training=False):

    w = create_variable(name='conv_w', shape=filter_shape)

    # convolution, no need for bias due to BN'mean calc
    conv_out = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')

    bn_out = batch_norm(conv_out, training=training)

    activation = tf.nn.relu(bn_out)

    return activation


def bn_relu_conv(x, filter_shape, stride, training=False):

    bn_out = batch_norm(x, training=training)

    activation = tf.nn.relu(bn_out)

    w = create_variable(name='conv_w', shape=filter_shape)

    # convolution, no need for bias due to BN'mean calc
    conv_out = tf.nn.conv2d(activation, w, strides=[1, stride, stride, 1], padding='SAME')

    return conv_out


'''Variant ResNet without BN layer'''


def basic_block_variant(x, out_depth, stride, first_resblk=False):

    input_depth = x.get_shape().as_list()[-1]

    with tf.variable_scope('convA'):

        if first_resblk:
            conv1 = conv2d(x, [3, 3, input_depth, out_depth], 1, padding='SAME')
        else:
            conv1 = conv_relu(x, [3, 3, input_depth, out_depth], stride)

    with tf.variable_scope("convB"):

        conv2 = conv2d(conv1, [3, 3, out_depth, out_depth], 1, padding='SAME')

    block_out = shortcut(x, conv2)

    return block_out


def bottleneck_variant(x, out_depth, stride, first_resblk=False):

    input_depth = x.get_shape().as_list()[-1]

    with tf.variable_scope('convA'):

        if first_resblk:
            conv1 = conv2d(x, [1, 1, input_depth, out_depth], 1, padding='SAME')
        else:
            conv1 = conv_relu(x, [1, 1, input_depth, out_depth], stride)

    with tf.variable_scope("convB"):

        conv2 = conv_relu(conv1, [3, 3, out_depth, out_depth], 1)

    with tf.variable_scope("convC"):

        conv3 = conv_relu(conv2, [1, 1, out_depth*4, out_depth], 1)

    conv_block_out = shortcut(x, conv3)

    return conv_block_out


def conv_relu(x, filter_shape, stride, padding='SAME'):

    w = create_variable(name='conv_w', shape=filter_shape)

    b = create_variable(name='conv_b', shape=[filter_shape[-1]], initializer=tf.zeros_initializer())

    xw = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)

    xw_b = tf.nn.bias_add(xw, b)

    # leak relu for regression problem
    activation = tf.nn.leaky_relu(xw_b, alpha=0.2)

    return activation


'''Basic High Level API'''


def shortcut(x, res):

    input_dim = x.get_shape().as_list()
    res_dim = res.get_shape().as_list()

    stride_h = int(round(input_dim[1] / res_dim[1]))
    stride_w = int(round(input_dim[2] / res_dim[2]))

    increase_dim = (input_dim[3] == res_dim[3])

    skip_connection = x

    if stride_h > 1 or stride_w > 1 or increase_dim:

        filter_shape = [1, 1, input_dim[3], res_dim[3]]

        w = create_variable(name='shortcut_w', shape=filter_shape)

        skip_connection = tf.nn.conv2d(x, w, strides=[1, stride_h, stride_w, 1], padding='VALID')

    final_out = skip_connection + res

    return final_out


def conv2d(x, filter_shape, stride, padding='SAME', name='conv', verbose=False):

    w = create_variable(name + '_w', shape=filter_shape)

    b = create_variable(name + '_b', shape=[filter_shape[-1]], initializer=tf.zeros_initializer())

    xw = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)

    xw_b = tf.nn.bias_add(xw, b)

    if verbose:
        print("> %s layer: knum=%d, ksize=%d, stride=%d, output_shape=%s"
              % (name, filter_shape[-1], filter_shape[0], stride, str(xw_b.get_shape())))

    return xw_b


def dense(x, num_outputs, use_relu, name='fc', verbose=False):

    num_inputs = x.get_shape().as_list()[-1]

    w = create_variable(name + '_w', shape=[num_inputs, num_outputs])

    b = create_variable(name + '_b', shape=[num_outputs], initializer=tf.zeros_initializer())

    act = tf.nn.xw_plus_b(x, w, b, name=name + '_logits')

    # regression last layer don't use this layer
    if use_relu:
        act = tf.nn.relu(act, name=name + '_relu')

    if verbose:
        print("> %s layer: num_in=%d, num_out=%d, output_shape=%s"
              % (name, num_inputs, num_outputs, str(act.get_shape())))

    return act


def flatten(x):

    nums = np.prod(x.get_shape().as_list()[1:])  # C*W*H

    return tf.reshape(x, [-1, nums])  # samples * (C*W*H)


def batch_norm(x, training=False, decay=0.9, eps=1e-3, name='bn'):

    x = tf.layers.batch_normalization(x, training=training, momentum=decay)

    return x


def create_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True,
                    weight_decay=0.0005, loss=tf.nn.l2_loss):
    # weight decay is default choice, if don't use please modify as 'weight_decay=None'

    new_var = tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable)

    if weight_decay:
        w_loss = loss(new_var) * weight_decay
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w_loss)

    return new_var


def max_pool(x, ksize, stride, padding):

    out = tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    return out


def lrn(x, radius, alpha, beta, bias=1.0):

    out = tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

    return out


def relu(x):

    activation = tf.nn.relu(x)

    return activation


def leak_relu(x):

    activation = tf.nn.leaky_relu(x, alpha=0.2)

    return activation


def batch_norm_ll(x, training=True, eps=1e-3, decay=0.9, affine=True, name='bn'):

    depth = x.get_shape().as_list()[-1]

    pop_mean = create_variable(name + '_pop_mean', shape=[depth], initializer=tf.zeros_initializer(), trainable=False)
    pop_var = create_variable(name + '_pop_var', shape=[depth], initializer=tf.ones_initializer(), trainable=False)

    def mean_var_with_update():
        axes = list(range(len(x.get_shape()) - 1))
        batch_mean, batch_var = tf.nn.moments(x, axes=axes, name='moments')
        with tf.control_dependencies([assign_moving_average(pop_mean, batch_mean, decay),
                                      assign_moving_average(pop_var, batch_var, decay)]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, variance = tf.cond(tf.cast(training, tf.bool), mean_var_with_update, lambda: (pop_mean, pop_var))

    if affine:
        beta = tf.get_variable(name + '_beta', shape=[depth], initializer=tf.zeros_initializer())
        gamma = tf.get_variable(name + '_gamma', shape=[depth], initializer=tf.ones_initializer())

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
    else:
        x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)

    return x

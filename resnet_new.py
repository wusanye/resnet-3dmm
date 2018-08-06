#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-08-04
#############################################

"""This is implementation of ResNet model."""

import tensorflow as tf
from nn_layers import residual_block, conv_bn_relu, max_pool, dense


class ResNet(object):

    def __init__(self, x, output_dim, block, block_num_list):

        self.predicts = ResNet.forward(x, block, output_dim, block_num_list)

    @staticmethod
    def forward(x, block, output_dim, block_num_list):

        layers = []

        with tf.variable_scope('conv0'):
            conv0 = conv_bn_relu(x, [7, 7, 3, 64], 2)
            pool0 = max_pool(conv0, ksize=3, stride=2, padding='SAME')
            layers.append(pool0)

        layers.append(make_conv_block(block, 1, layers[-1], 64, block_num_list[0], stride=1))

        layers.append(make_conv_block(block, 2, layers[-1], 128, block_num_list[1], stride=2))

        layers.append(make_conv_block(block, 3, layers[-1], 256, block_num_list[2], stride=2))

        layers.append(make_conv_block(block, 4, layers[-1], 512, block_num_list[3], stride=2))

        with tf.variable_scope('avg_pool_dense'):
            avg_pool = tf.reduce_mean(layers[-1], [1, 2])
            fc = dense(avg_pool, output_dim, use_relu=False)
            layers.append(fc)

        return layers[-1]


def make_conv_block(block, block_id, x, output_depth, num_residual_blocks, stride):

    strides = [stride] + [1]*(num_residual_blocks - 1)

    first = (stride == 1)

    layers = [x]

    for no in range(num_residual_blocks):

        with tf.variable_scope('conv%d_%d' % (block_id, no)):

            out = residual_block(block, layers[-1], output_depth, strides[no], is_first_miniblock=first)

            layers.append(out)

    return layers[-1]


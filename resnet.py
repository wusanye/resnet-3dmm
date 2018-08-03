#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
#############################################

"""This is implementation of ResNet model."""

import tensorflow as tf
import numpy as np


class ResNet(object):
    def __init__(self, x, output_dim):
        self.verbose = True
        self.x = x
        self.output_dim = output_dim

    def ResNet101(self):
        self.build_net("Bottleneck", [3, 4, 23, 3])

    def ResNet34(self):
        self.build_net("BasicBlock", [3, 4, 6, 3])

    def ResNet50(self):
        self.build_net("Bottleneck", [3, 4, 6, 3])

    def ResNet18(self):
        self.build_net("BasicBlock", [2, 2, 2, 2])

    def build_net(self, block_type, repeats):
        """build the ResNet"""
        filter_shape = [7, 7, 3, 64]
        stride_shape = [2, 2]

        conv1 = self._conv_bn_relu(self.x, filter_shape, stride_shape)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        block = pool1
        filter_num = 64
        for i, r in enumerate(repeats):
            block = self.residual_block(block, block_type, filter_num, r, (i == 0))
            filter_num *= 2

        block_shape = block.get_shape().as_list()
        pool2 = tf.nn.avg_pool(block, ksize=[1, block_shape[1], block_shape[2], 1], strides=[1, 1, 1, 1], padding='SAME')

        flattened = self._flatten(pool2)
        out = self._fc_layer(flattened, self.output_dim)

        self.predicts = out

    def residual_block(self, x, block_type, filter_num, repeats, is_first_layer=False):
        """bulid the residual block with repeated block"""
        for i in range(repeats):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            is_first = (is_first_layer and i == 0)

            if block_type == "BasicBlock":
                x = self.basic_block(x, filter_num, init_strides, is_first)
            elif block_type == "Bottleneck":
                x = self.bottleneck(x, filter_num, init_strides, is_first)

        return x

    def basic_block(self, x, filter_num, init_strides=(1, 1), is_first_block_of_first_layer=False):
        """
        3x3 convolution block for ResNet with layers < 50.
        implemented with pre-activation ResNet.
        """
        filter_shape = [3, 3, x.get_shape().as_list()[-1], filter_num]
        stride_shape = list(init_strides)
        w = self._get_weight_variable(filter_shape)

        if is_first_block_of_first_layer:
            # do not need to repeat bn->relu since we just did the bn->relu->maxpool
            conv1 = tf.nn.conv2d(x, w, strides=[1, stride_shape[0], stride_shape[1], 1], padding='SAME')
        else:
            conv1 = self._bn_relu_conv(x, filter_shape, stride_shape)

        filter_shape1 = [3, 3, conv1.get_shape().as_list()[-1], filter_num]
        stride_shape1 = [1, 1]  # stride is 1 except the first conv-layer of first block of first layer
        res = self._bn_relu_conv(conv1, filter_shape1, stride_shape1)

        block_out = self._shortcut(x, res)

        return block_out

    def bottleneck(self, x, filter_num, init_strides=(1, 1), is_first_block_of_first_layer=False):
        """
        1x1->3x3->1x1 convolution block for ResNet with layers >= 50.
        implemented with pre-activation ResNet.
        """
        filter_shape = [1, 1, x.get_shape().as_list()[-1], filter_num]
        stride_shape = list(init_strides)
        w = self._get_weight_variable(filter_shape)

        if is_first_block_of_first_layer:
            # do not need to repeat bn->relu since we just did the bn->relu->maxpool
            conv1 = tf.nn.conv2d(x, w, strides=[1, stride_shape[0], stride_shape[1], 1], padding='SAME')
        else:
            conv1 = self._bn_relu_conv(x, filter_shape, stride_shape)

        # stride is 1 except the first conv-layer of first block of first layer
        stride_shape_inner = [1, 1]

        # middle conv layer is 3x3
        filter_shape1 = [3, 3, conv1.get_shape().as_list()[-1], filter_num]
        conv2 = self._bn_relu_conv(conv1, filter_shape1, stride_shape_inner)

        # final conv-layer of block is 4x
        filter_shape2 = [1, 1, conv2.get_shape().as_list()[-1], filter_num*4]
        res = self._bn_relu_conv(conv2, filter_shape2, stride_shape_inner)

        block_out = self._shortcut(x, res)

        return block_out

    def _shortcut(self, x, res):
        """shortcut between input and residual block, add input and residual"""
        # Expand channels of shortcut to match residual's
        # Stride should be int if network architecture is configured correctly
        # To match residual's height and width
        input_shape = x.get_shape().as_list()
        res_shape = res.get_shape().as_list()
        stride_h = int(round(input_shape[1] / res_shape[1]))
        stride_w = int(round(input_shape[2] / res_shape[2]))
        equal_channel = (input_shape[3] == res_shape[3])

        shortcut = x
        # # 1 X 1 conv if shape is different, else identity.
        if stride_h > 1 or stride_w > 1 or not equal_channel:
            filter_shape = [1, 1, input_shape[3], res_shape[3]]
            w = self._get_weight_variable(filter_shape)
            shortcut = tf.nn.conv2d(x, w, strides=[1, stride_h, stride_w, 1], padding='VALID')

        out = shortcut + res

        return out

    def _conv_bn_relu(self, x, filter_shape, stride_shape):
        """conv -> bn -> relu"""
        # init filter weights
        w = self._get_weight_variable(filter_shape, name='weights')

        # convolution, no need for bias due to BN'mean calc
        conv_out = tf.nn.conv2d(x, w, strides=[1, stride_shape[0], stride_shape[1], 1], padding='SAME')

        activation = self._bn_relu(conv_out)

        return activation

    def _bn_relu_conv(self, x, filter_shape, stride_shape):
        """bn -> relu -> conv"""
        # init filter weights
        activation = self._bn_relu(x)

        w = self._get_weight_variable(filter_shape, name='weights')

        # convolution, no need for bias due to BN'mean calc
        conv_out = tf.nn.conv2d(activation, w, strides=[1, stride_shape[0], stride_shape[1], 1], padding='SAME')

        return conv_out

    def _bn_relu(self, input):
        """bn -> relu block"""
        depth = input.get_shape().as_list()[-1]

        # mean and variance calc on batch-height-width dimension
        mean, var = tf.nn.moments(input, axes=[0, 1, 2])
        beta = tf.Variable(tf.zeros([depth]), name='beta')
        gamma = self._get_weight_variable([depth], name='gamma')

        bn_out = tf.nn.batch_norm_with_global_normalization(input, mean, var, beta, gamma, 0.001,
                                                            scale_after_normalization=True)

        out = tf.nn.relu(bn_out)

        return out

    @staticmethod
    def _fc_layer(x, num_out, activation=None):
        """fully connected layer"""
        num_in = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1), name="fc_weights")
        bias = tf.Variable(tf.zeros([num_out, ]), name="fc_biases")
        output = tf.nn.xw_plus_b(x, weight, bias)
        if activation:
            output = activation(output)

        return output

    @staticmethod
    def _flatten(x):
        """flatten layer"""
        nums = np.prod(x.get_shape().as_list()[1:])  # C*W*H
        return tf.reshape(x, [-1, nums])  # samples * (C*W*H)

    @staticmethod
    def _get_weight_variable(shape, name=None):
        """weight initialization"""
        # shape: [height, width, in_depth, out_depth]
        initial = tf.truncated_normal(shape, stddev=0.1)

        return tf.Variable(initial, name=name)






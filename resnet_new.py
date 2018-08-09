#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-08-04
#############################################

"""This is implementation of ResNet model."""

import cv2
import numpy as np
import tensorflow as tf
from nn_layers import residual_block, conv_bn_relu, max_pool, dense


class ResNet(object):

    def __init__(self, image_size, output_dim, block, block_num_list):
        self.output_dim = output_dim
        self.block = block
        self.block_num_list = block_num_list
        self.images = tf.placeholder(tf.float32, [None, *image_size], name='data_feed')
        self.predicts = ResNet.forward(self.images, block, output_dim, block_num_list)

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
            fc = dense(avg_pool, output_dim, use_relu=False, name='output')
            layers.append(fc)

        return layers[-1]

    @staticmethod
    def inference(model_path, input_file):

        image = cv2.cvtColor(cv2.imread(input_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        image_batch = np.expand_dims(image, axis=0)

        # ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.import_meta_graph(model_path + '.meta')

        graph = tf.get_default_graph()
        # with tf.get_default_graph() as graph:
        x = graph.get_tensor_by_name("data_feed:0")
        out = graph.get_tensor_by_name("output_fc:0")

        with tf.Session() as sess:
            saver.restore(sess, model_path)

            preds = sess.run(out, feed_dict={x: image_batch})

        return preds

    def predict(self, model_path, input_file):

        image = cv2.cvtColor(cv2.imread(input_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.

        image_batch = np.expand_dims(image, axis=0)

        # model = ResNet(self.output_dim, self.block, self.block_num_list)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, model_path)

            predicts = sess.run(self.predicts, feed_dict={self.images: image_batch})

        return predicts


def make_conv_block(block, block_id, x, output_depth, num_residual_blocks, stride):

    strides = [stride] + [1]*(num_residual_blocks - 1)

    first = (stride == 1)

    layers = [x]

    for no in range(num_residual_blocks):

        with tf.variable_scope('conv%d_%d' % (block_id, no)):

            out = residual_block(block, layers[-1], output_depth, strides[no], is_first_miniblock=first)

            layers.append(out)

    return layers[-1]


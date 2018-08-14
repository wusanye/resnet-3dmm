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
        self.training = tf.placeholder(tf.bool, name='training')
        self.images = tf.placeholder(tf.float32, [None, *image_size], name='input_data')
        self.predicts = ResNet.forward(self.images, block, output_dim, block_num_list, self.training)

    @staticmethod
    def forward(x, block, output_dim, block_num_list, training):

        with tf.variable_scope('conv0'):
            x = conv_bn_relu(x, [7, 7, 3, 64], 2)
            x = max_pool(x, ksize=3, stride=2, padding='SAME')

        x = make_conv_block(block, 1, x, 64,  block_num_list[0], 1, training)

        x = make_conv_block(block, 2, x, 128, block_num_list[1], 2, training)

        x = make_conv_block(block, 3, x, 256, block_num_list[2], 2, training)

        x = make_conv_block(block, 4, x, 512, block_num_list[3], 2, training)

        with tf.variable_scope('dense'):
            x = tf.reduce_mean(x, [1, 2])
            x = dense(x, output_dim, use_relu=False, name='output')

        return x

    @staticmethod
    def inference(model_path, input_file):

        image = cv2.cvtColor(cv2.imread(input_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        image_batch = np.expand_dims(image, axis=0)

        # ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.import_meta_graph(model_path + '.meta')

        graph = tf.get_default_graph()
        # with tf.get_default_graph() as graph:
        x = graph.get_tensor_by_name("input_data:0")
        out = graph.get_tensor_by_name("dense/output_logits:0")
        training = graph.get_tensor_by_name("training:0")

        with tf.Session() as sess:
            saver.restore(sess, model_path)

            preds = sess.run(out, feed_dict={x: image_batch, training: 0})

        return preds

    def predict(self, model_path, input_file):

        image = cv2.cvtColor(cv2.imread(input_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.

        image_batch = np.expand_dims(image, axis=0)

        # model = ResNet(self.output_dim, self.block, self.block_num_list)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, model_path)

            predicts = sess.run(self.predicts, feed_dict={self.images: image_batch, self.training: 0})

        return predicts


def make_conv_block(block, block_id, x, out_depth, num_res_blks, stride, training):

    strides = [stride] + [1]*(num_res_blks - 1)

    first = (stride == 1)

    layers = [x]

    for no in range(num_res_blks):

        with tf.variable_scope('conv%d_%d' % (block_id, no)):

            out = residual_block(block, layers[-1], out_depth, strides[no], (first and no == 0), training)

            layers.append(out)

    return layers[-1]


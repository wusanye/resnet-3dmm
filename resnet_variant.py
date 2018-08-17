#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-08-04
#############################################

"""This is implementation of Variant ResNet model without BN"""

import cv2
import numpy as np
import tensorflow as tf
from utils import DataGenerator
from nn_layers import conv_relu, max_pool, dense, leaky_relu, dropout


class ResNet(object):

    def __init__(self, image_size, output_dim, block, block_num_list):
        self.output_dim = output_dim
        self.block = block
        self.block_num_list = block_num_list
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.images = tf.placeholder(tf.float32, [None, *image_size], name='input_data')
        self.predicts = ResNet.forward(self.images, block, output_dim, block_num_list, self.keep_prob)

    @staticmethod
    def forward(x, block, output_dim, block_num_list, keep_prob):

        with tf.variable_scope('conv0'):
            x = conv_relu(x, [7, 7, 3, 64], 2)
            x = max_pool(x, ksize=3, stride=2, padding='SAME')

        x = stack(block, 1, x, 64,  block_num_list[0], 1)

        x = stack(block, 2, x, 128, block_num_list[1], 2)

        x = stack(block, 3, x, 256, block_num_list[2], 2)

        x = stack(block, 4, x, 512, block_num_list[3], 2)

        with tf.variable_scope('dense5'):
            x = tf.reduce_mean(x, [1, 2])
            # x = dense(x, 1024, activation=leaky_relu, name='dense')
            # x = dropout(x, keep_prob=keep_prob)
            x = dense(x, output_dim, name='output')

        return x

    @staticmethod
    def inference(model_path, input_file):

        image = DataGenerator.parse_data(input_file)

        image_batch = tf.expand_dims(image, axis=0)

        # ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.import_meta_graph(model_path + '.meta')

        graph = tf.get_default_graph()
        # with tf.get_default_graph() as graph:
        x = graph.get_tensor_by_name("input_data:0")
        out = graph.get_tensor_by_name("dense5/output_logits:0")

        with tf.Session() as sess:
            saver.restore(sess, model_path)

            image_batch = sess.run(image_batch)

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


def stack(block, block_id, x, out_depth, num_res_blks, stride):

    strides = [stride] + [1]*(num_res_blks - 1)

    first = (stride == 1)

    layers = [x]

    for no in range(num_res_blks):

        with tf.variable_scope('conv%d_%d' % (block_id, no)):

            out = block(layers[-1], out_depth, strides[no], (first and no == 0))

            layers.append(out)

    return layers[-1]


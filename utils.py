#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
#############################################

"""This is implementation of batch preprocess."""

import os
import glob
import numpy as np
import collections
import tensorflow as tf
# from os import listdir
# from os.path import isfile, join
from datetime import datetime
from tensorflow.data import Dataset
from tensorflow.data import Iterator
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

TRAIN_SET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

DataMember = collections.namedtuple("DataMember", ['init_op', 'num_examples', 'batch_size'])

DataFamily = collections.namedtuple("DataFamily", ['train', 'val', 'next_batch'])


class DataGenerator(object):

    def __init__(self, txt_file, image_size, output_dim, mode, batch_size, shuffle=True,
                 buffer_size=10000):

        self.txt_file = txt_file
        self.output_dim = output_dim
        self.image_size = image_size

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.float32)

        # slice first dimension of tensor to create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)
        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls=8)
        else:
            raise ValueError("Invalid mode '%s'." % mode)

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self):
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip()  # strip the last '\n'
                self.img_paths.append(item)
                # read y data, stored in .txt
                num_list = read_txt(item[:-3] + "txt")
                self.labels.append(num_list)

    def _shuffle_lists(self):
        img_paths = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(img_paths[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, img_file, label):

        img_normed = self.parse_data(img_file)

        label_normed = self.parse_label(label, self.image_size)

        return img_normed, label_normed

    def _parse_function_inference(self, img_file, label):

        img_normed = self.parse_data(img_file)

        label_normed = self.parse_label(label, self.image_size)

        return img_normed, label_normed

    @staticmethod
    def parse_data(img_file):
        # load and pre-process the image
        img_string = tf.read_file(img_file)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [224, 224])
        img_normed = tf.divide(tf.cast(img_resized, tf.float32), 255.)

        return img_normed

    @staticmethod
    def parse_label(label, image_size):
        # label pre-process
        label_con = label_norm(image_size)
        label_con = convert_to_tensor(label_con, dtype=tf.float32)
        label_normed = tf.divide(label, label_con)

        return label_normed


def label_norm(image_size):
    f = np.load('bfm09/std_shape_exp.npz')
    shape_std, exp_std = f['shape_ev'], f['exp_ev']
    rest_std = np.array([1, 1, 1, image_size, image_size, image_size/224.], dtype=np.float32)
    label_con = np.concatenate((shape_std, exp_std, rest_std))

    return label_con


def read_txt(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    num_list = []
    for line in lines:
        tmp = list(map(float, line.strip('\n').strip().split(' ')))
        num_list += tmp
    return num_list


def asym_l2_loss(predicts, truth):
    """Implementation of Asymmetric Euclidean Loss
    Args:
        predicts: predicted output, dim (batch, output_len)
        truth: ground truth
    Returns:
        Asymmetric Euclidean Loss over the batch
    """
    lambda1 = tf.constant(1/4, dtype=tf.float32)
    lambda2 = tf.constant(3/4, dtype=tf.float32)

    gamma_plus = tf.abs(truth)
    gamma_pplus = tf.sign(truth) * predicts
    gamma_max = tf.maximum(gamma_plus, gamma_pplus)

    # over_estimate = lambda1 * tf.square(tf.norm(gamma_plus - gamma_max, axis=1))
    # under_estimate = lambda2 * tf.square(tf.norm(gamma_pplus - gamma_max, axis=1))

    over_estimate = lambda1 * tf.reduce_mean(tf.square(gamma_plus - gamma_max), axis=1)
    under_estimate = lambda2 * tf.reduce_mean(tf.square(gamma_pplus - gamma_max), axis=1)

    return tf.reduce_mean(over_estimate + under_estimate)


def l2_loss(predicts, truths):

    loss = tf.reduce_mean(tf.square(predicts - truths))

    return loss


def optimize_loss(train_optimizer, lr_rate, predicts, truths, summary=True):

    with tf.name_scope("Asymmetric_L2_Loss"):
        loss = asym_l2_loss(predicts, truths)
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss += tf.add_n(reg_loss)

    var_list = [v for v in tf.trainable_variables()]

    # # optimizer.minimize
    with tf.name_scope("train"):
        optimizer = train_optimizer(learning_rate=lr_rate)
        grads_and_vars = optimizer.compute_gradients(loss, var_list)
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # batch norm update

    if summary:
        # # statistic summary
        for var in var_list:
            tf.summary.histogram(var.name, var)

        for gradient, var in grads_and_vars:
            tf.summary.histogram(var.name + '/gradient', gradient)

    tf.summary.scalar('loss', loss)

    return train_op, update_op, loss


def load_data_sets(image_size, output_dim, batch_size, train_list, val_list):

    # Place data loading and pre-processing on cpu
    with tf.device('/cpu:0'):
        train_data = DataGenerator(train_list, image_size[0], output_dim, 'training', batch_size, shuffle=True)
        val_data = DataGenerator(val_list, image_size[0], output_dim, 'inference', batch_size, shuffle=False)

    # Create an reinitializable iterator given the data structure
    iterator = Iterator.from_structure(train_data.data.output_types, train_data.data.output_shapes)
    next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    train_init_op = iterator.make_initializer(train_data.data)
    val_init_op = iterator.make_initializer(val_data.data)

    train_member = DataMember(init_op=train_init_op, num_examples=train_data.data_size, batch_size=batch_size)
    val_member = DataMember(init_op=val_init_op, num_examples=val_data.data_size, batch_size=batch_size)

    return DataFamily(train=train_member, val=val_member, next_batch=next_batch)


def load_data_set():
    pass


def train(train_op, update_op, loss, feeds, dataset, epochs, saver, logdir, display_step=100):

    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(logdir + '/train')
    dev_writer = tf.summary.FileWriter(logdir + '/dev')

    train_batches = np.ceil(dataset.train.num_examples / dataset.train.batch_size)
    val_batches = np.ceil(dataset.val.num_examples / dataset.val.bnatch_size)

    train_init_op = dataset.train.init_op
    val_init_op = dataset.val.init_op

    next_batch = dataset.next_batch

    # # begin training
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        sess.run(tf.global_variables_initializer())

        train_writer.add_graph(sess.graph)

        print("{} begin training...".format(datetime.now().strftime(TIME_FORMAT)))

        for epoch in range(epochs):

            print("{} epoch number: {}".format(datetime.now().strftime(TIME_FORMAT), epoch + 1))

            # Training here
            sess.run(train_init_op)

            for batch in range(train_batches):

                x_batch, y_batch = sess.run(next_batch)

                feed_dict = dict(zip([*feeds], [x_batch, y_batch, True]))
                _, _, obj_loss = sess.run([train_op, update_op, loss], feed_dict=feed_dict)

                rel_loss = obj_loss / np.mean(np.square(y_batch))

                if batch % display_step == 0:
                    print("{}, epoch: {}, loss: {:.8f}, rel loss: {:.9f}"
                          .format(datetime.now().strftime(TIME_FORMAT), epoch + 1, obj_loss, rel_loss))

                    feed_dict = dict(zip([*feeds], [x_batch, y_batch, False]))
                    s = sess.run(merged_summary, feed_dict=feed_dict)
                    train_writer.add_summary(s, epoch * train_batches + batch)

            # Validating here
            print("{} begin validation".format(datetime.now().strftime(TIME_FORMAT)))

            sess.run(val_init_op)

            loss_list = []

            for b in range(val_batches):

                x_batch, y_batch = sess.run(next_batch)

                feed_dict = dict(zip([*feeds], [x_batch, y_batch, False]))
                obj_loss = sess.run([loss], feed_dict=feed_dict)

                loss_list.append(obj_loss)

                if b % display_step == 0:
                    s = sess.run(merged_summary, feed_dict=feed_dict)
                    dev_writer.add_summary(s, epoch * train_batches + b)

            average_loss = np.mean(loss_list)
            print("{}, epoch: {}, val loss: {:.8f}".format(datetime.now().strftime(TIME_FORMAT), epoch + 1, average_loss))

            # # checkpoint save
            checkpoint_name = os.path.join(logdir + '/ckpts', 'model_epoch' + str(epoch + 1) + '.ckpt')
            saver.save(sess, checkpoint_name)
            print("{} model checkpoint saved at {}".format(datetime.now().strftime(TIME_FORMAT), checkpoint_name))


def get_file_list(files_path, save_file):
    # 1st method
    file_list = glob.glob(os.path.join(files_path, '*.png'))

    # 2nd method
    # only_files = [f for f in listdir(files_path) if isfile(join(files_path, f))]

    with open(save_file, 'w') as f:
        f.write('\n'.join(file_list))  # write(a single string) while writelines(list of string)






#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
#############################################

"""This is implementation of batch preprocess."""

import os
import numpy as np
import _pickle as pickle
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

TRAIN_SET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class DataGenerator(object):
    """
    Data set generator, with tensorflow Dataset and iterator
    """
    def __init__(self, txt_file, image_size, output_dim, mode, batch_size, shuffle=True,
                 buffer_size=10000):
        """Create a new ImageDataGenerator.
        Receives a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensorFlow datasets, that can be used to train
        e.g. a convolution neural network.
        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Whether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        """
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
        """Read the content of the text file and store it into lists."""
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
        """Conjoined shuffling of the list of paths and labels."""
        img_paths = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(img_paths[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, img_file, label):
        """input parser for samples of the training set."""

        img_normed, label_normed = self.parse_data(img_file, label, self.image_size)

        return img_normed, label_normed

    def _parse_function_inference(self, img_file, label):
        """input parser for samples of the validation/test set."""

        img_normed, label_normed = self.parse_data(img_file, label, self.image_size)

        return img_normed, label_normed

    @staticmethod
    def parse_data(image, label, image_size):
        # label pre-process
        label_con = label_norm(image_size)
        label_con = convert_to_tensor(label_con, dtype=tf.float32)
        label_normed = tf.divide(label, label_con)

        # load and pre-process the image
        img_string = tf.read_file(image)
        img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        # img_resized = tf.image.resize_images(img_decoded, [224, 224])
        img_normed = tf.divide(tf.cast(img_decoded, tf.float32), 255.)

        return img_normed, label_normed


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

    over_estimate = lambda1 * tf.reduce_sum(tf.square(gamma_plus - gamma_max), axis=1)
    under_estimate = lambda2 * tf.reduce_sum(tf.square(gamma_pplus - gamma_max), axis=1)

    return tf.reduce_mean(over_estimate + under_estimate)


def label_norm(image_size):

    f = np.load('bfm09/std_shape_exp.npz')
    shape_std, exp_std = f['shape_ev'], f['exp_ev']
    rest_std = np.array([1, 1, 1, image_size, image_size, 1], dtype=np.float32)
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


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def load_data(dataset_dir, class_num):
    if not os.path.isdir(dataset_dir):
        print("Error: input data set directory is invalid!")

    train_batches = [os.path.join(dataset_dir, "data_batch_"+str(i)) for i in range(1, 6)]

    xlist, ylist = [], []
    for batch in train_batches:
        d = unpickle(batch)
        xlist.append(d[b"data"])
        ylist.append(d[b"labels"])

    x_train = np.vstack(xlist)
    y_train = np.vstack(ylist)

    with open(os.path.join(dataset_dir, "test_batch"), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        x_test, y_test = d[b"data"], d[b"labels"]

    y_train = np.reshape(y_train, (-1))
    y_test = np.array(y_test).reshape(-1)

    y_train = np.eye(class_num)[y_train]
    y_test = np.eye(class_num)[y_test]

    # shuffle data set
    shuffle_idx = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_idx, :]
    y_train = y_train[shuffle_idx, :]

    # shuffle data set
    shuffle_idx = np.random.permutation(len(x_test))
    x_test = x_test[shuffle_idx, :]
    y_test = y_test[shuffle_idx, :]

    return x_train, y_train, x_test, y_test


def normalize_data(data, labels, size):
    # mean value in all train data set
    mu = np.mean(data, axis=0)  # calc on the batch dimension
    mu = mu.reshape(1, -1)

    # std value in all train data set
    sigma = np.std(data, axis=0)
    sigma = sigma.reshape(1, -1)

    # normalization
    data = data - mu
    data = data / sigma

    # data = data.reshape(-1, depth, height, width)
    # data = data.transpose([0, 2, 3, 1])

    data = data.astype(np.float32)
    labels = labels.astype(np.float32)

    return data, labels


def get_batch(data, labels, batch_size, batch_no):
    return data[batch_size*(batch_no - 1): batch_size*batch_no: 1, :], \
           labels[batch_size*(batch_no - 1): batch_size*batch_no: 1, :]


def split_list(txt_file):

    with open(txt_file, 'r') as f:
        lines = f.readlines()
        img_paths = [line for line in lines]

    permutation = np.random.permutation(len(img_paths))

    img_list = [img_paths[i] for i in permutation]

    train_list = img_list[:64000]
    val_list = img_list[64000:]

    with open('training.list', 'w+') as f1:
        for line in train_list:
            f1.write(line)

    with open('validating.list', 'w+') as f2:
        for line in val_list:
            f2.write(line)









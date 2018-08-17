#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
#############################################

"""THIS IS THE IMPLEMENTATION OF INFERENCE PROCESS"""

import h5py
import numpy as np
import tensorflow as tf
from resnet_new import ResNet
from nn_layers import basic_block
from utils import DataGenerator
from tensorflow.data import Iterator


def predict(model_path, list_file, image_size, output_dim, batch_size):

    with tf.device('/cpu:0'):
        val_data = DataGenerator(list_file, image_size, output_dim, mode='inference', batch_size=batch_size, shuffle=False)

        # Create an reinitializable iterator given the data structure
        iterator = Iterator.from_structure(val_data.data.output_types, val_data.data.output_shapes)
        next_batch = iterator.get_next()

    val_init_op = iterator.make_initializer(val_data.data)

    val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

    model = ResNet([image_size, image_size, 3], output_dim, basic_block, [3, 4, 6, 3])

    saver = tf.train.Saver()

    with tf.Session() as sess:

        saver.restore(sess, model_path)

        sess.run(val_init_op)

        loss_list = []

        for i in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)

            preds = sess.run(model.predicts, feed_dict={model.images: img_batch, model.training: False})

            loss = asym_l2_loss(preds, label_batch)

            loss_list.append(loss)

        average_loss = sum(loss_list) / len(loss_list)

        return average_loss


def asym_l2_loss(predicts, truth):
    """Implementation of Asymmetric Euclidean Loss
    Args:
        predicts: predicted output, dim (batch, output_len)
        truth: ground truth
    Returns:
        Asymmetric Euclidean Loss over the batch
    """
    lambda1 = 1 / 4.
    lambda2 = 3 / 4.

    gamma_plus = np.abs(truth)
    gamma_pplus = np.sign(truth) * predicts
    gamma_max = np.maximum(gamma_plus, gamma_pplus)

    over_estimate = lambda1 * np.mean(np.square(gamma_plus - gamma_max), axis=1)
    under_estimate = lambda2 * np.mean(np.square(gamma_pplus - gamma_max), axis=1)

    total_error = over_estimate + under_estimate

    return np.mean(total_error)


def l2_loss(predicts, truths):

    loss = np.mean(np.square(predicts - truths))

    return loss


def inference(model_path, list_file, image_size, output_dim, batch_size):

    with tf.device('/cpu:0'):
        val_data = DataGenerator(list_file, image_size, output_dim, mode='inference', batch_size=batch_size, shuffle=False)

        # Create an reinitializable iterator given the data structure
        iterator = Iterator.from_structure(val_data.data.output_types, val_data.data.output_shapes)
        next_batch = iterator.get_next()

    val_init_op = iterator.make_initializer(val_data.data)

    val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

    saver = tf.train.import_meta_graph(model_path + '.meta')
    graph = tf.get_default_graph()
    # with tf.get_default_graph() as graph:
    x = graph.get_tensor_by_name("input_data:0")
    out = graph.get_tensor_by_name("dense5/output_logits:0")
    # training = graph.get_tensor_by_name("training:0")

    with tf.Session() as sess:

        saver.restore(sess, model_path)

        sess.run(val_init_op)

        loss_list = []

        for i in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)

            np.savetxt('label.txt', label_batch)

            preds = sess.run(out, feed_dict={x: img_batch})  # , training: False})

            np.savetxt('pred.txt', preds)

            loss = asym_l2_loss(preds, label_batch)

            loss_list.append(loss)

        average_loss = sum(loss_list) / len(loss_list)

        # print(loss_list)

        return average_loss


if __name__ == '__main__':
    data_list = './list/performance.list'
    image_size = 450
    output_dim = 185
    batch_size = 1
    ckpt_path = "./experiment/resnet-variant-displayal2/model_epoch{}.ckpt".format(100)  #

    average_loss = inference(ckpt_path, data_list, image_size, output_dim, batch_size)

    # print(loss_list)
    print(average_loss)







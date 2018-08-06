#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
#############################################

"""THIS IS THE IMPLEMENTATION OF MAIN TRAINING PROCESS"""

import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
# from resnet import ResNet
from resnet_new import ResNet
from nn_layers import basic_block
from utils import DataGenerator, asymmetric_euclidean_loss, read_txt
from datetime import datetime
from tensorflow.data import Iterator
from tensorflow.python import  debug as tf_debug

"""Configuration Part"""
# Path to the text files for the training/val/test set
train_list = "./train.list"
val_list = "val.list"
test_list = "test.list"

# Learning parameters
learning_rate = 0.001
lr_annealing = 0.5
lr_epochs = 10
# 10 epochs for seeing the effect on the whole
epochs = 50
# 256 for first try, try x2 or x0.5
batch_size = 128  # 128

# Network parameters
output_dim = 185

# Display frequency
display_step = 20

filewriter_path = "tmp/tensorboard"

if not os.path.exists(filewriter_path):
    os.makedirs(filewriter_path)

"""Main Part of Training"""
# Place data loading and pre-processing on cpu
with tf.device('/cpu:0'):
    train_data = DataGenerator(train_list, mode='training', batch_size=batch_size, shuffle=True)
    # val_data = DataGenerator(val_list, mode='inference', batch_size=batch_size, shuffle=False)

    # Create an reinitializable iterator given the data structure
    iterator = Iterator.from_structure(train_data.data.output_types, train_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
train_init_op = iterator.make_initializer(train_data.data)
# val_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [batch_size, output_dim])

# Initialize model
# ResNet = ResNet(x, output_dim)
# ResNet.ResNet50()
model = ResNet(x, output_dim, basic_block, [3, 4, 6, 3])

predicts = model.predicts

var_list = [v for v in tf.trainable_variables()]

with tf.name_scope("Asymmetric_Euclidean_Loss"):
    loss = asymmetric_euclidean_loss(predicts, y)

'''
with tf.name_scope("train"):
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# for gradient, var in gradients:
#    tf.summary.histogram(var.name + '/gradient', gradient)

for var in var_list:
    tf.summary.histogram(var.name, var)

tf.summary.scalar('asymmetric_euclidean_loss', loss)

merged_summary = tf.summary.merge_all()
'''

writer = tf.summary.FileWriter(filewriter_path)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

train_batches_per_epoch = int(np.floor(train_data.data_size / batch_size))
# val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

f = open("loss.dat", 'w+')

with tf.Session() as sess:

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    sess.run(tf.global_variables_initializer())

    writer.add_graph(sess.graph)

    print("{} Start training...".format(datetime.now()))

    for epoch in range(epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        # Training here
        sess.run(train_init_op)

        for batch in range(train_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)
            
            _, loss_value, pred = sess.run([optimizer, loss, predicts], feed_dict={x: img_batch, y: label_batch})

            relative_loss = loss_value / np.mean(np.square(np.linalg.norm(label_batch, axis=1)))

            if batch % display_step == 0:
                print("{}, Loss: {}, Relative loss: {}".format(datetime.now(), loss_value, relative_loss))
                f.write("{}, Loss: {}, Relative loss: {}".format(datetime.now(), loss_value, relative_loss))
                # s = sess.run(merged_summary, feed_dict={x: img_batch, y: label_batch})
                # writer.add_summary(s, epoch * train_batches_per_epoch + batch)

        # annealing lr every 5 epochs by half
        if epoch % lr_epochs == 0:
            learning_rate *= lr_annealing

        """
        # Validating here
        print("{} Start validation".format(datetime.now()))

        sess.run(val_init_op)

        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)

            val_loss_value = sess.run([loss], feed_dict={x: img_batch, y: label_batch})

            print("{} Validation Loss = {}".format(datetime.now(), val_loss_value))
        """

f.close()

























#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
#############################################

"""THIS IS THE IMPLEMENTATION OF MAIN TRAINING PROCESS"""

import os
import shutil
import numpy as np
import tensorflow as tf
from resnet_variant import ResNet
from nn_layers import basic_block_variant, bottleneck_variant
from utils import DataGenerator, asym_l2_loss, optimize_loss, l2_loss
from datetime import datetime
from tensorflow.data import Iterator
# from tensorflow.python import debug as tf_debug

'''configuration part'''
# Path to the text files for the training/val/test set
image_size = [450, 450, 3]
train_list = "./list/train.list"
val_list = "./list/dev.list"
test_list = "./test.list"

# 10 epochs for seeing the effect on the whole
epochs = 120
# 256 for first try, try x2 or x0.5
batch_size = 32  # 128
# Network parameters
output_dim = 185
# Display frequency (print/#batch)
display_step = 400

writer_path = "visualization"
checkpoint_path = "checkpoints"

if os.path.exists(writer_path):
    shutil.rmtree(writer_path)
os.makedirs(writer_path)

'''main part of training'''
# Place data loading and pre-processing on cpu
with tf.device('/cpu:0'):
    train_data = DataGenerator(train_list, image_size[0], output_dim, mode='training', batch_size=batch_size, shuffle=True)
    val_data = DataGenerator(val_list, image_size[0], output_dim, mode='inference', batch_size=batch_size, shuffle=False)

    # Create an reinitializable iterator given the data structure
    iterator = Iterator.from_structure(train_data.data.output_types, train_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
train_init_op = iterator.make_initializer(train_data.data)
val_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
y = tf.placeholder(tf.float32, [None, output_dim], name='data_label')

# Initialize model
model = ResNet([224, 224, 3], output_dim, basic_block_variant, [3, 4, 6, 3])

predicts = model.predicts
x, keep_prob = model.images, model.keep_prob

with tf.name_scope("Asymmetric_L2_Loss"):
    l2_loss = asym_l2_loss(predicts, y)
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = l2_loss + tf.add_n(reg_loss)

var_list = [v for v in tf.trainable_variables()]

# # annealing the learning rate with different decay curve
global_step = tf.Variable(0, name='global_step', trainable=False)
boundaries = [10*2000, 20*2000, 40*2000, 60*2000]
lr_value = [0.001, 0.005, 0.001, 0.0005, 0.0001]
lr_rate = tf.train.piecewise_constant(global_step, boundaries, lr_value)

# # optimizer.minimize
with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr_rate, momentum=0.9, use_nesterov=True)
    grads_and_vars = optimizer.compute_gradients(loss, var_list)
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

# # statistic summary
for var in var_list:
    tf.summary.histogram(var.name, var)

for gradient, var in grads_and_vars:
    tf.summary.histogram(var.name + '/gradient', gradient)

tf.summary.scalar('lr', lr_rate)
tf.summary.scalar('l2 loss', l2_loss)
tf.summary.scalar('total loss', loss)

merged_summary = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(writer_path + '/train')
dev_writer = tf.summary.FileWriter(writer_path + '/dev')

saver = tf.train.Saver(max_to_keep=epochs)

train_batches_per_epoch = int(np.floor(train_data.data_size / batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# # begin training
with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    sess.run(tf.global_variables_initializer())

    train_writer.add_graph(sess.graph)

    print("{} Start training...".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    for epoch in range(epochs):

        print("{} Epoch number: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch+1))

        # Training here
        sess.run(train_init_op)

        for batch in range(train_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)

            _, _, obj_loss = sess.run([train_op, loss, l2_loss], feed_dict={x: img_batch, y: label_batch, keep_prob: 0.5})

            rel_loss = obj_loss / np.mean(np.square(label_batch))

            if batch % display_step == 0:
                print("{}, Epoch: {}, Loss: {:.8f}, Rel loss: {:.9f}"
                      .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch+1, obj_loss, rel_loss))

                s = sess.run(merged_summary, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})
                train_writer.add_summary(s, epoch * train_batches_per_epoch + batch)

        # Validating here
        print("{} Start validation".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        sess.run(val_init_op)

        loss_list = []

        for b in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(next_batch)

            total_loss, obj_loss = sess.run([loss, l2_loss], feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})

            # rel_loss = loss_value / np.mean(np.square(label_batch))

            loss_list.append(obj_loss)

            if b % display_step == 0:

                s = sess.run(merged_summary, feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})
                dev_writer.add_summary(s, epoch * train_batches_per_epoch + b)

        average_loss = np.mean(loss_list)
        print("{}, Epoch: {}, Val Loss: {:.8f}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                       epoch + 1, average_loss))

        # # checkpoint save
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        print("{} Model checkpoint saved at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), checkpoint_name))



















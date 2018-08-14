#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
#############################################

"""THIS IS THE IMPLEMENTATION OF MAIN TRAINING PROCESS"""

import os
import shutil
import numpy as np
import tensorflow as tf
from resnet_new import ResNet
from nn_layers import basic_block
from utils import DataGenerator, asym_l2_loss, optimize_loss
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
epochs = 40
# 256 for first try, try x2 or x0.5
batch_size = 32  # 128
# Network parameters
output_dim = 185
# Display frequency (print/#batch)
display_step = 500

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
model = ResNet([224, 224, 3], output_dim, basic_block, [3, 4, 6, 3])

predicts = model.predicts
x, training = model.images, model.training

with tf.name_scope("Asymmetric_L2_Loss"):
    loss = asym_l2_loss(predicts, y)
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss += tf.add_n(reg_loss)

var_list = [v for v in tf.trainable_variables()]

# # annealing the learning rate with different decay curve
global_step = tf.Variable(0, name='global_step', trainable=False)
boundaries = [5*2000, 10*2000, 40*2000, 60*2000]
lr_value = [0.0001, 0.00005, 0.00002, 0.00001, 0.000005]
lr_rate = tf.train.piecewise_constant(global_step, boundaries, lr_value)

# # optimizer.minimize
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate)
    grads_and_vars = optimizer.compute_gradients(loss, var_list)
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # batch norm update

# # statistic summary
for var in var_list:
    tf.summary.histogram(var.name, var)

for gradient, var in grads_and_vars:
    tf.summary.histogram(var.name + '/gradient', gradient)

tf.summary.scalar('lr', lr_rate)
tf.summary.scalar('loss', loss)

merged_summary = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(writer_path + '/train')
dev_writer = tf.summary.FileWriter(writer_path + '/dev')

saver = tf.train.Saver(max_to_keep=epochs)

train_batches_per_epoch = int(np.floor(train_data.data_size / batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

# # begin training
with open("loss.dat", 'w+') as f:

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

                _, _, loss_value = sess.run([train_op, update_op, loss], feed_dict={x: img_batch, y: label_batch, training: True})

                rel_loss = loss_value / np.mean(np.square(np.linalg.norm(label_batch, axis=1)))

                if batch % display_step == 0:
                    print("{}, Epoch: {}, Loss: {:.8f}, Rel loss: {:.9f}"
                          .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch+1, loss_value, rel_loss))
                    f.write("{}, Epoch: {}, Loss: {:.8f}, Rel loss: {:.9f}\n"
                            .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch+1, loss_value, rel_loss))

                    s = sess.run(merged_summary, feed_dict={x: img_batch, y: label_batch, training: False})
                    train_writer.add_summary(s, epoch * train_batches_per_epoch + batch)

            # Validating here
            print("{} Start validation".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            sess.run(val_init_op)

            for _ in range(val_batches_per_epoch):

                img_batch, label_batch = sess.run(next_batch)

                loss_value = sess.run(loss, feed_dict={x: img_batch, y: label_batch, training: False})

                rel_loss = loss_value / np.mean(np.square(np.linalg.norm(label_batch, axis=1)))

                if _ % display_step == 0:
                    print("{}, Epoch: {}, Val Loss: {:.8f}, Rel loss: {:.9f}"
                          .format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch+1, loss_value, rel_loss))

                    s = sess.run(merged_summary, feed_dict={x: img_batch, y: label_batch, training: False})
                    dev_writer.add_summary(s, epoch * train_batches_per_epoch + _)

            # # checkpoint save
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)
            print("{} Model checkpoint saved at {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), checkpoint_name))



















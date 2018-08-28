#############################################
# Copyright (c) 2018-present
# written by Kai Wu on 2018-07-31
#############################################

"""THIS IS THE IMPLEMENTATION OF MAIN TRAINING PROCESS"""

import os
import shutil
import tensorflow as tf
from collections import OrderedDict
from resnet_new import ResNet
from nn_layers import basic_block
from utils import asym_l2_loss, optimize_loss, l2_loss, load_data_sets, train
# from tensorflow.python import debug as tf_debug


os.environ['CUDA_VISIBLE_DEVICES'] = '5'

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

logdir = "train_logs"

if os.path.exists(logdir):
    shutil.rmtree(logdir)
os.makedirs(logdir)

'''main part of training'''
# # load train & validation data
data_family = load_data_sets(image_size[0], output_dim, batch_size, train_list, val_list)

# TF placeholder for graph input and output
y = tf.placeholder(tf.float32, [None, output_dim], name='data_label')

# Initialize model
model = ResNet([224, 224, 3], output_dim, basic_block, [3, 4, 6, 3])

predicts = model.predicts
x, training = model.images, model.training

with tf.name_scope("Asymmetric_L2_Loss"):
    obj_loss = asym_l2_loss(predicts, y)
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = obj_loss + tf.add_n(reg_loss)

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
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # batch norm update

# # statistic summary
for var in var_list:
    tf.summary.histogram(var.name, var)

for gradient, var in grads_and_vars:
    tf.summary.histogram(var.name + '/gradient', gradient)

tf.summary.scalar('lr', lr_rate)
tf.summary.scalar('l2 loss', obj_loss)
tf.summary.scalar('total loss', loss)

saver = tf.train.Saver(max_to_keep=epochs)

feed_dict = OrderedDict.fromkeys([x, y, training])

train(train_op, update_op, [loss, obj_loss], feed_dict, data_family, epochs, saver, logdir)















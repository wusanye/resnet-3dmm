import numpy as np
import tensorflow as tf
from utils import DataGenerator
from tensorflow.data import Iterator
import cv2
from matplotlib import pyplot as plt
from keras_applications import resnet50

batch_size = 1
train_file = "train.list"
data = DataGenerator(train_file, mode='training', batch_size=batch_size, shuffle=True)

# create an reinitializable iterator given the dataset structure
iterator = Iterator.from_structure(data.data.output_types, data.data.output_shapes)
next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(data.data)

with tf.Session() as sess:

    # Initialize iterator with the training dataset
    sess.run(training_init_op)

    try:
        while True:
            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)
            print(img_batch.shape, label_batch.shape)
            # plt.imshow(img_batch[0])
            # plt.show()

    except tf.errors.OutOfRangeError:
        print("end!")





import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, './model/')
from prepare_input import extract_patches
from model import *

#----------------VARIABLES----------------
patch_height = 48
patch_width = 48
channels = 1
num_patches = 190000
inside_FOV = False

num_epochs = 150
learn_rate = 0.01
batch_size = 32

images = tf.placeholder(tf.float32, shape=(None, patch_height, patch_width, channels))
grnd_truths = tf.placeholder(tf.float32, shape=(None, patch_height, patch_width, channels))

#after running the model with xavier initializers try He init (Variance Scaling) for ReLU!!
weights = {
    'w1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w2': tf.get_variable('W1', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w3': tf.get_variable('W2', shape=(3, 3, 32, 64), initializer=tf.contrib.layers.xavier_initializer()),
    'w4': tf.get_variable('W3', shape=(3, 3, 64, 64), initializer=tf.contrib.layers.xavier_initializer()),
    'w5': tf.get_variable('W4', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),
    'w6': tf.get_variable('W5', shape=(3, 3, 128, 128), initializer=tf.contrib.layers.xavier_initializer()),
    'w7': tf.get_variable('W6', shape=(3, 3, 128, 64), initializer=tf.contrib.layers.xavier_initializer()),
    'w8': tf.get_variable('W7', shape=(3, 3, 64, 64), initializer=tf.contrib.layers.xavier_initializer()),
    'w9': tf.get_variable('W8', shape=(3, 3, 64, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w10': tf.get_variable('W9', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w11': tf.get_variable('W10', shape=(1, 1, 32, 2), initializer=tf.contrib.layers.xavier_initializer())
}

biases = {
    'b1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'b4': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'b5': tf.get_variable('B4', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'b6': tf.get_variable('B5', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'b7': tf.get_variable('B6', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'b8': tf.get_variable('B7', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'b9': tf.get_variable('B8', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b10': tf.get_variable('B9', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b11': tf.get_variable('B10', shape=(2), initializer=tf.contrib.layers.xavier_initializer())
}
#-----------------------------------------

#patches, patches_gt = extract_patches(patch_height, patch_width, num_patches, inside_FOV)

model = unet(images, weights, biases)

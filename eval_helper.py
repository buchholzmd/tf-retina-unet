import os
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, './model/')
from file_processing import *
from prepare_input import *

def get_data(dataset_path, data_type):
    '''
        Function to get training/testing data

        Args:
          dataset_path: string, path to write/read pre-processed image data
          data_type: string, indicate whether training or testing data
          
        Returns:
          patches: numpy.array, 4D nunmpy array containing training/testing patches
          patches_gt: numpy.array, 4D nunmpy array containing training/testing ground truth patches
    '''
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
        patches, patches_gt = extract_patches(patch_height, patch_width, num_patches, inside_FOV)
        patches_gt = prepare_grnd_truths(patches_gt)
    
        write(patches, dataset_path + "training_images.hdf5")
        write(patches_gt, dataset_path + "training_ground_truths.hdf5")
    
    else:
        patches = load(dataset_path + "training_images.hdf5")
        patches_gt = load(dataset_path + "training_ground_truths.hdf5")
    
    return patches, patches_gt
    

def compute_accuracy(predictions, labels):
    '''
        Function to calculate accuracy of predictions

        Args:
          predictions: tensorflow.Tensor, input tensor
          labels: tensorflow.Tensor, filter tensor
          
        Returns:
          numpy.sum: tensorflow.Tensor, fraction of correct number of predictions for a given batch
    '''
    num_correct = np.sum(np.argmax(predictions, axis=1)==np.argmax(labels, axis=1))
    return num_correct/labels.shape[0]

def visualize_performance(gradients):
    '''
        Function to get summaries for TensorBoard

        Args:
          gradients: tensorflow.Tensor, tensor of gradients and their variances to check for exploding/vanishing gradients
          
        Returns:
          tf.summary.merge: tensorflow.summary, merged accuracy and loss summary for TensorBoard
          grad_norm_summary: tensorflow.summary.scalar, summary of gradient norms for TensorBoard
    '''
    with tf.name_scope('performance'):
        loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
        loss_summary = tf.summary.scalar('loss', loss_ph)
        
        accuracy_ph = tf.placeholder(tf.float32, shape=None, name='accuracy_summary')
        accuracy_summary = tf.summary.scalar('accuracy', accuracy_ph)
        
    last = len(gradients)-1
    grads, var = gradients[last]
    
    with tf.name_scope('gradients'):
        grad_norm = tf.sqrt(tf.reduce_mean(grads**2))
        grad_norm_summary = tf.summary.scalar('gradient_norm', grad_norm)
    
    return tf.summary.merge([loss_summary, accuracy_summary]), grad_norm_summary, loss_ph, accuracy_ph
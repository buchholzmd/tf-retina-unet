import os
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, './model/')
from file_processing import *
from prepare_input import *

#----------------VARIABLES----------------
patch_height = 48
patch_width = 48
channels = 1
num_patches = 190000
stride = 5
inside_FOV = False

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
    if data_type == "train":
        
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
    
            train_patches, train_patches_gt = extract_train_patches(patch_height, patch_width, num_patches, inside_FOV)
            train_patches_gt = prepare_grnd_truths(train_patches_gt)
    
            write(train_patches, dataset_path + "training_patches.hdf5")
            write(train_patches_gt, dataset_path + "training_patches_gts.hdf5")
    
        else:
            train_patches = load(dataset_path + "training_images.hdf5")
            train_patches_gt = load(dataset_path + "training_ground_truths.hdf5")
    
        return train_patches, train_patches_gt
    
    elif data_type == "test":
        
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
    
            new_size, test_patches, test_patches_gt = extract_test_patches(patch_height, patch_width, stride)
            
            write_tuple(new_size, dataset_path + "new_size.pickle")
            write(test_patches, dataset_path + "testing_patches.hdf5")
            write(test_patches_gt, dataset_path + "testing_patches_gts.hdf5")
    
        else:
            new_size = load_tuple(dataset_path + "new_size.pickle")
            test_patches = load(dataset_path + "testing_patches.hdf5")
            test_patches_gt = load(dataset_path + "testing_patches_gts.hdf5")
    
        return new_size, test_patches, test_patches_gt
    
    else:
        print("Please specify whether training or testing!")
        sys.exit()

def compute_accuracy(predictions, labels):
    '''
        Function to calculate accuracy of predictions

        Args:
          predictions: tensorflow.Tensor, input tensor
          labels: tensorflow.Tensor, filter tensor
          
        Returns:
          numpy.sum: tensorflow.Tensor, fraction of correct number of predictions for a given batch
    '''
    num_correct = np.sum(np.argmax(predictions, axis=2)==np.argmax(labels, axis=2))
    return num_correct/(labels.shape[0] * labels.shape[1])

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

def conv_to_imgs(predictions, patch_height, patch_width, stride):
    '''
        Function to convert predictions/classifications into image patches

        Args:
          predictions: numpy.array, outputted softmax logits for binary classification of pixels
          patch_height: int, height of patches
          patch_width: int, width of patches
          stride: int, stride from window when extracting patches
          
        Returns:
          np.reshape: numpy.array, 4D nunmpy array containing prediction patches (threshold for classification is 0.5)
    '''
    assert(len(predictions.shape)==3)
    assert(predictions.shape[2]==2)
    
    print("---------------------------------------")
    print("Converting predictions to image data")
    print("---------------------------------------\n")
    
    images = np.empty((predictions.shape[0], predictions.shape[1]))
    for i in range(predictions.shape[0]):
        for pixel in range(predictions.shape[1]):
            if predictions[i, pixel, 1] >= 0.5:
                images[i, pixel] = 1
            else:
                images[i, pixel] = 0

    return np.reshape(images, (images.shape[0], patch_height, patch_width, 1))

def recombine_patches(pred_patches, new_size, stride):
    '''
        Function to recombine prediction patches into a full image

        Args:
          pred_patches: numpy.array, patches of logits for binary classification of pixels
          new_size: tuple, height and width of new image size (including padding for uniformity in dimensions)
          stride: int, stride from window when extracting patches
          
        Returns:
          avg_prob: numpy.array, 4D nunmpy array containing average pixel classification across all patches, recombined into a full image
    '''
    assert(len(pred_patches.shape)==4)
    assert(pred_patches.shape[3]==1)
    
    height = new_size[0]
    width = new_size[1]
    patch_height = pred_patches.shape[1]
    patch_width = pred_patches.shape[2]
    
    patches_per_img = ((height-patch_height)//stride+1)*((width-patch_width)//stride+1)
    ##--check that number of patches per image evenly distributed over total images--
    assert(pred_patches.shape[0]%patches_per_img==0)
    
    num_imgs = pred_patches.shape[0]//patches_per_img
    
    ##--array of probability sums--
    prob_arr = np.zeros((num_imgs, height, width, pred_patches.shape[3]))
    sum_arr = np.zeros((num_imgs, height, width, pred_patches.shape[3]))
    
    count = 0
    
    print("---------------------------------------")
    print("Recombining patches")
    print("---------------------------------------\n")
    
    for image in range(num_imgs):
        for i in range((height-patch_height)//stride+1):
            for j in range((width-patch_width)//stride+1):
                prob_arr[image, i*stride:(i*stride)+patch_height, j*stride:(j*stride)+patch_width, :] += pred_patches[count]
                sum_arr[image, i*stride:(i*stride)+patch_height, j*stride:(j*stride)+patch_width, :] += 1
                count += 1
    
    assert(count==pred_patches.shape[0])
    avg_prob = prob_arr/sum_arr
    assert(np.max(avg_prob) <= 1.0)
    assert(np.min(avg_prob) >= 0.0)
                
    return avg_prob

def remove_border(pred_images, masks):
    '''
        Function to remove padding provided to account for overlap by patch extraction

        Args:
          pred_patches: numpy.array, patches of logits for binary classification of pixels
          new_size: tuple, height and width of new image size (including padding for uniformity in dimensions)
          stride: int, stride from window when extracting patches
          
        Returns:
          avg_prob: numpy.array, 4D nunmpy array containing average pixel classification across all patches, recombined into a full image
    '''
    assert(len(pred_images.shape)==4)
    assert(pred_images.shape[3]==1)
    
    height = pred_images.shape[1]
    width = pred_images.shape[2]
    
    print("---------------------------------------")
    print("Removing borders")
    print("---------------------------------------\n")
    
    for image in range(pred_images.shape[0]):
        for i in range(height):
            for j in range(width):
                if not inside_masks(image, i, j, masks):
                    pred_images[image, i, j, :] = 0.0
    
def inside_masks(image, i, j, masks):
    '''
        Function to determine whether a given pixel coordinate (j, i) is within the field of view for its respective border mask

        Args:
          image: numpy.array: 4D array of image data
          i: int, row(y) dimension for given pixel
          j: int, column(x) dimension for given pixel
          masks: numpy.array: 4D array of border mask
          
        Returns:
          True/False: bool, whether inside field of view or not
    '''
    assert(len(masks.shape)==4)
    assert(masks.shape[3]==1)
    
    #--check if pixels are outside of original size
    if(i >= masks.shape[1] or j >= masks.shape[2]):
        return False
    
    #--check if pixels are black--
    if(masks[image, i, j, 0] > 0):
        return True
    
    return False

def group(data):
    '''
        Function to group images together for outputting/saving predictions with ground truths

        Args:
          data: numpy.array: 4D array of image data
          
        Returns:
          img: numpy.array, 3D array of grouped images
    '''
    group = []
    
    print("---------------------------------------")
    print("Grouping images")
    print("---------------------------------------\n")
    
    for i in range(int(data.shape[0])):
        group.append(data[i])
        
    img = group[0]
    for i in range(1, len(group)):
        img = np.concatenate((img, group[i]), axis=0)
    return img

def prediction_in_FOV(pred_images, test_gts, masks):
    '''
        Function to get only the predictions within the field of view

        Args:
          pred_images: numpy.array: 4D array of image predictions
          test_gts: numpy.array: 4D array of image ground truths
          masks: numpy.array: 4D array of image border masks
          
        Returns:
          new_preds: numpy.array, 4D array of array of predicted pixel classification only within field of view
          new_gts: numpy.array, 4D array of array of ground truths of pixels only within field of view
    '''
    assert(len(pred_images.shape)==4 and len(test_gts.shape)==4)
    assert(pred_images.shape[0] == test_gts.shape[0] and \
           pred_images.shape[1] == test_gts.shape[1] and \
           pred_images.shape[2] == test_gts.shape[2] and \
           pred_images.shape[3] == test_gts.shape[3])
    
    height = pred_images.shape[1]
    width = pred_images.shape[2]
    
    new_preds = []
    new_gts = []
    
    for image in range(pred_images.shape[0]):
        for i in range(height):
            for j in range(width):
                if inside_masks(image, i, j, masks):
                    new_preds.append(pred_images[image, i, j, :])
                    new_gts.append(test_gts[image, i, j, :])
    new_preds = np.asarray(new_preds)
    new_gts = np.asarray(new_gts)
    
    return new_preds, new_gts
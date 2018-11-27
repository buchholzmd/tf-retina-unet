import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#---------------PATH VARIABLES---------------
##train##
orig_imgs_train = "./DRIVE/training/images/"
grnd_truths_train = "./DRIVE/training/1st_manual/"
masks_train = "./DRIVE/training/mask/"

##test##
orig_imgs_test = "./DRIVE/test/images/"
grnd_truths_test = "./DRIVE/test/1st_manual/"
masks_test = "./DRIVE/test/mask/"
#--------------------------------------------

num_imgs = 20
height, width, channels = 584, 565, 3
data_path = "./dataset"

def get_data(image_dir, grnd_truth_dir, masks_dir, dataset="null"):
    '''
        This function gets the training/test data from its respective directory

        Args:
          image_dir: string, path to training/test images
          grnd_truth_dir: string, path to training/test ground truths
          masks_dir: string, path to training/test border masks
          dataset: string, flag to indicate whether training or test data
          
        Returns:
          raw_data: numpy.array, 4D array of training/test image data
          grnd_truths: numpy.array, 3D array of training/test ground truths
          masks: numpy.array, 3D array of training/test border masks
    '''
    for path, subdirs, files in os.walk(image_dir):
        #get original image data
        raw_data = np.array([mpimg.imread(image_dir+files[i]) for i in range(len(files))])
        #get corresponding ground truths
        grnd_truths = np.array([mpimg.imread(grnd_truth_dir+files[i][0:2]+"_manual1.gif") for i in range(len(files))])
        #get corresponding border masks
        if(dataset == "train"):
            masks = np.array([mpimg.imread(masks_dir+files[i][0:2]+"_training_mask.gif") for i in range(len(files))])
        elif(dataset == "test"):
            masks = np.array([mpimg.imread(masks_dir+files[i][0:2]+"_test_mask.gif") for i in range(len(files))])
        else:
            print("Must specify whether training or test dataset!")


    #verify dimesnions of data
    assert(raw_data.shape == (num_imgs, height, width, channels))
    assert(grnd_truths.shape == (num_imgs, height, width))
    assert(masks.shape == (num_imgs, height, width))

    return raw_data, grnd_truths, masks

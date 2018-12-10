import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def augment_data(data):
        '''
        This function performs all data augmentation on the input arrays

        Args:
          data: numpy.array, contains 20 images and respective pixel data
          RGB channels are last dimension of input
        Returns:
          data: numpy.array, contains 20 pre-processed images
        '''
##        check that input is 4D array
        assert(len(data.shape) == 4)
##        check that each image has 3 feature channels
        assert(data.shape[3] == 3)
        data = grayscale_conv(data)
        data = clahe(data)
        data = standardize(data)
        data = normalize(data)
        data = gamma_correct(data, 1.2)

        return data

def grayscale_conv(data):
        '''
        This function performs grayscale conversion on the input arrays

        Args:
          data: numpy.array, contains 20 images and respective pixel data
          RGB channels are last dimension of input
        Returns:
          data: numpy.array, contains 20 grayscale images
        '''
##        check that input is 4D array
        assert(len(data.shape) == 4)
##        check that each image has 3 feature channels
        assert(data.shape[3] == 3)
        new_img = np.empty(shape=(20, 584, 565, 1))
        luminance = np.array([0.2126, 0.7152, 0.0722])
        for i in range(len(data)):
                new_img[i] = np.expand_dims(np.dot(data[i,:,:,:3], luminance), axis=2)

        return new_img

def standardize(data):
        '''
        This function performs standardization on the input arrays to give form gaussian distribution

        Args:
          data: numpy.array, contains 20 images and respective pixel data
        Returns:
          data: numpy.array, contains 20 normalized images
        '''
##        check that input is 4D array
        assert(len(data.shape) == 4)
##        check that each image has 1 feature channel
        assert(data.shape[3] == 1)
        img_std = np.std(data)
        img_mean = np.mean(data)

        data = data - img_mean
        data = data / img_std

        return data

def normalize(data):
        '''
        This function performs normalization on the input arrays to scale output range

        Args:
          data: numpy.array, contains 20 images and respective pixel data
        Returns:
          data: numpy.array, contains 20 normalized images
        '''
##        check that input is 4D array
        assert(len(data.shape) == 4)
##        check that each image has 1 feature channel
        assert(data.shape[3] == 1)
        for i in range(len(data)):
                img_min = np.amin(data[i])
                img_max = np.amax(data[i])
                alpha = 1 / (img_max - img_min)

                data[i] = data[i] - img_min
                data[i] = data[i] * alpha
        
        return data

def clahe(data):
        '''
        This function performs contrast limited adaptive histogram equalization on the input arrays
        Histogram equalization is applied to 5x5 local neighborhoods of pixels
        If any bins of a local region's histogram is abive a specified contrast limit,
        those bins are clipped and uniformly distributed to other bins
        Bilinear interpolation is applied to remove artifact from overlapping equalization in tile borders

        Args:
          data: numpy.array, contains 20 images and respective pixel data
        Returns:
          data: numpy.array, contains 20 CLAHE images
        '''
##        check that input is 4D array
        assert(len(data.shape) == 4)
##        check that each image has 1 feature channel
        assert(data.shape[3] == 1)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized_data = np.empty(data.shape)
        for i in range(len(data)):
                clahe_img = clahe.apply(np.array(data[i], dtype=np.uint8))
                equalized_data[i] = np.expand_dims(clahe_img, axis=2)

        return equalized_data

def gamma_correct(data, gamma):
        '''
        This function performs gamma decoding on the input arrays

        Args:
          data: numpy.array, contains 20 images and respective pixel data
          gamma: float, value of gamma for correction
        Returns:
          data: numpy.array, contains 20 gamma corrected images
        '''
##        check that input is 4D array
        assert(len(data.shape) == 4)
##        check that each image has 1 feature channel
        assert(data.shape[3] == 1)

        #adjust pixel data using gamma parameter
        data = data/255
        data = data ** (1.0/gamma)
        data = data * 255

        return data
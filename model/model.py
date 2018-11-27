import numpy as np
import tensorflow as tf

def unet(images, filters, biases):
    '''
        This function gets the tensorflow u-net model 

        Args:
          images: tensorflow.Tensor, input tensor
          filters: tensorflow.Tensor, filter tensor
          bias: tensorflow.Tensor, bias tensor
          
        Returns:
          model: tensorflow.estimator, 4D array of training/test image data
    '''
    #------------------------
    #Contracting Path/Encoder
    #------------------------
    conv1 = conv2d(images, filters['w1'], biases['b1'])
    conv1 = dropout(conv1, 0.2)
    conv1 = conv2d(conv1, filters['w2'], biases['b2'])
    pool1 = maxpool2d(conv1, (2,2))
    #
    conv2 = conv2d(pool1, filters['w3'], biases['b3'])
    conv2 = dropout(conv2, 0.2)
    conv2 = conv2d(conv2, filters['w4'], biases['b4'])
    pool2 = maxpool2d(conv2, (2,2))
    #
    conv3 = conv2d(pool2, filters['w5'], biases['b5'])
    conv3 = dropout(conv3, 0.2)
    conv3 = conv2d(conv3, filters['w6'], biases['b6'])
    #----------------------
    #Expansive Path/Decoder
    #----------------------
    up1 = upsample(conv3, (24,24))
    up1 = tf.concat(3, [conv2, up1])
    conv4 = conv2d(up1, filters['w7'], biases['b7'])
    conv4 = dropout(conv4, 0.2)
    conv4 = conv2d(conv4, filters['w8'], biases['b8'])
    #
    up2 = upsample(conv4, (24,24))
    up2 = tf.concat(3, [conv1, up2])
    conv5 = conv2d(up2, filters['w9'], biases['b9'])
    conv5 = dropout(conv5, 0.2)
    conv5 = conv2d(conv5, filters['w10'], biases['b10'])
    #
    model = conv2d(conv5, filters['w11'], biases['b11'])

    return model
    
def conv2d(images, filters, bias, stride=(1,1)):
    '''
        Wrapper function for 2D convolution operation

        Args:
          images: tensorflow.Tensor, input tensor
          filters: tensorflow.Tensor, filter tensor
          bias: tensorflow.Tensor, bias tensor
          stride: int, stride for sliding window of filter
          
        Returns:
          tensorflow.nn.relu: tensorflow.Tensor, relu activated output Tensor for layer
    '''
    images = tf.nn.conv2d(images, filters, strides=[1, stride[0], stride[1], 1], padding='SAME')
    images = tf.nn.bias_add(images, bias)
    return tf.nn.relu(images)

def maxpool2d(images, kernel_size=(2,2)):
    '''
        Wrapper function for 2D max pooling operation

        Args:
          images: tensorflow.Tensor, input tensor
          kernel_size: int, stride for pooling filter size
          
        Returns:
          tensorflow.nn.max_pool: tensorflow.Tensor, max pooled output Tensor
    '''
    return tf.nn.max_pool(images, ksize=[1, kernel_size[0], kernel_size[1], 1], strides=[1, kernel_size[0], kernel_size[1], 1], padding='SAME')

def dropout(images, dropout_rate):
    '''
        Wrapper function for Dropout operation

        Args:
          images: tensorflow.Tensor, input tensor
          dropout_rate: int, dropout rate
          
        Returns:
          tensorflow.nn.dropout: tensorflow.Tensor, output Tensor with dropout
    '''
    return tf.nn.dropout(images, 1-dropout_rate)

##experiment with interpolation method!! Blinear, bicubic etc...
##also experiment with transposed_conv instead
def upsample(images, size=(2,2)):
    '''
        Wrapper function for Dropout operation

        Args:
          images: tensorflow.Tensor, input tensor
          size: int, upsampling factor
          
        Returns:
          tensorflow.nn.dropout: tensorflow.Tensor, output Tensor with dropout
    '''
    return tf.image.resize_images(images, size=[size[0], size[1]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

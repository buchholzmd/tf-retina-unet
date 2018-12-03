import numpy as np
import tensorflow as tf

#try batch normalization since fully convolutional
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
    orig_height = int(images.get_shape()[1])
    orig_width = int(images.get_shape()[2])

    conv1 = conv2d(images, filters['w1'], biases['b1'])
    conv1 = dropout(conv1, 0.2)
    conv1 = conv2d(conv1, filters['w2'], biases['b2'])
    up1 = upsample(conv1, (2,2))
    #
    conv2 = conv2d(up1, filters['w3'], biases['b3'])
    conv2 = dropout(conv2, 0.2)
    conv2 = conv2d(conv2, filters['w4'], biases['b4'])
    pool1 = maxpool2d(conv2, (2,2))
    #
    conv3 = conv2d(pool1, filters['w5'], biases['b5'])
    conv3 = dropout(conv3, 0.2)
    conv3 = conv2d(conv3, filters['w6'], biases['b6'])
    pool2 = maxpool2d(conv3, (2,2))
    #
    conv4 = conv2d(pool2, filters['w7'], biases['b7'])
    conv4 = dropout(conv4, 0.2)
    conv4 = conv2d(conv4, filters['w8'], biases['b8'])
    pool3 = maxpool2d(conv4, (2,2))
    #
    conv5 = conv2d(pool3, filters['w9'], biases['b9'])
    conv5 = dropout(conv5, 0.2)
    conv5 = conv2d(conv5, filters['w10'], biases['b10'])
    #----------------------
    #Expansive Path/Decoder
    #----------------------
    up2 = upsample(conv5, (2,2))
    up2 = tf.concat([up2, conv4], 3)
    conv6 = conv2d(up2, filters['w11'], biases['b11'])
    conv6 = dropout(conv6, 0.2)
    conv6 = conv2d(conv6, filters['w12'], biases['b12'])
    #
    up3 = upsample(conv6, (2,2))
    up3 = tf.concat([up3, conv3], 3)
    conv7 = conv2d(up3, filters['w13'], biases['b13'])
    conv7 = dropout(conv7, 0.2)
    conv7 = conv2d(conv7, filters['w14'], biases['b14'])
    #
    up4 = upsample(conv7, (2,2))
    up4 = tf.concat([up4, conv2], 3)
    conv8 = conv2d(up4, filters['w15'], biases['b15'])
    conv8 = dropout(conv8, 0.2)
    conv8 = conv2d(conv8, filters['w16'], biases['b16'])
    #
    pool4 = maxpool2d(conv8, (2,2))
    conv9 = conv2d(pool4, filters['w17'], biases['b17'])
    conv9 = dropout(conv9, 0.2)
    conv9 = conv2d(conv9, filters['w18'], biases['b18'])
    #
    conv10 = conv2d(conv9, filters['w19'], biases['b19'])
    conv10 = tf.reshape(conv10, [tf.shape(conv10)[0], 2, orig_height*orig_width])
    model = tf.transpose(conv10, perm=[0, 2, 1])

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
    return tf.nn.max_pool(images, ksize=[1, kernel_size[0], kernel_size[1], 1],
                          strides=[1, kernel_size[0], kernel_size[1], 1], padding='VALID')

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
    new_height = size[0]*int(images.get_shape()[1])
    new_width = size[1]*int(images.get_shape()[2])
    
    return tf.image.resize_images(images, size=[new_height, new_width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

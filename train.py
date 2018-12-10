import numpy as np
import tensorflow as tf
from sklearn import model_selection

from random import shuffle
from eval_helper import *
from model import *

#----------------VARIABLES----------------
patch_height = 48
patch_width = 48
channels = 1
num_patches = 190000
num_train_patches = 171000
num_valid_patches = 19000
inside_FOV = False

num_epochs = 150
learn_rate1 = 0.01
learn_rate2 = 0.001
momentum = 0.3
batch_size = 32
val_batch_size = 128
dataset_path = "./datasets/"
test_path = "./test/"
model_path = "./test/model.ckpt"

images = tf.placeholder(tf.float32, shape=(None, patch_height, patch_width, channels), name='Input')
grnd_truths = tf.placeholder(tf.float32, shape=(None, patch_height*patch_width, 2), name='Labels')

weights = {
    'w1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w2': tf.get_variable('W1', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w3': tf.get_variable('W2', shape=(3, 3, 32, 16), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w4': tf.get_variable('W3', shape=(3, 3, 16, 16), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w5': tf.get_variable('W4', shape=(3, 3, 16, 32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w6': tf.get_variable('W5', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w7': tf.get_variable('W6', shape=(3, 3, 32, 64), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w8': tf.get_variable('W7', shape=(3, 3, 64, 64), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w9': tf.get_variable('W8', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w10': tf.get_variable('W9', shape=(3, 3, 128, 128), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w11': tf.get_variable('W10', shape=(3, 3, 192, 64), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w12': tf.get_variable('W11', shape=(3, 3, 64, 64), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w13': tf.get_variable('W12', shape=(3, 3, 96, 32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w14': tf.get_variable('W13', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w15': tf.get_variable('W14', shape=(3, 3, 48, 16), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w16': tf.get_variable('W15', shape=(3, 3, 16, 16), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w17': tf.get_variable('W16', shape=(3, 3, 16, 32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w18': tf.get_variable('W17', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'w19': tf.get_variable('W18', shape=(1, 1, 32, 2), initializer=tf.contrib.layers.variance_scaling_initializer())
}

biases = {
    'b1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b3': tf.get_variable('B2', shape=(16), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b4': tf.get_variable('B3', shape=(16), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b5': tf.get_variable('B4', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b6': tf.get_variable('B5', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b7': tf.get_variable('B6', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b8': tf.get_variable('B7', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b9': tf.get_variable('B8', shape=(128), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b10': tf.get_variable('B9', shape=(128), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b11': tf.get_variable('B10', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b12': tf.get_variable('B11', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b13': tf.get_variable('B12', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b14': tf.get_variable('B13', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b15': tf.get_variable('B14', shape=(16), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b16': tf.get_variable('B15', shape=(16), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b17': tf.get_variable('B16', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b18': tf.get_variable('B17', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
    'b19': tf.get_variable('B18', shape=(2), initializer=tf.contrib.layers.variance_scaling_initializer())
}
#-----------------------------------------

##--get data--
patches, patches_gt = get_data(dataset_path, "train")

##--get model, loss and optimizer--
model = unet(images, weights, biases)

saver = tf.train.Saver()

predictions = tf.nn.softmax(model, name="predictions")
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=grnd_truths), name="loss")
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate1)
optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate1, momentum=momentum)
#optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate2, epsilon=1e-07)

##--compute gradients and minimize loss function--
gradients = optimizer.compute_gradients(cost)
min_cost = optimizer.minimize(cost)

##--get summaries--
performance, grad_norm_summary, loss_ph, accuracy_ph = visualize_performance(gradients)

##--configure GPU memory usage--
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

if not os.path.exists(test_path):
    os.makedirs(test_path)
    
init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

with tf.Session(config=config) as sess:
    ##--initialize variables/metric arrays--
    sess.run(init)
    sess.run(init_l)
    train_loss_arr = []
    val_loss_arr = []
    train_accuracy_arr = []
    val_accuracy_arr = []
    best_accuracy = 0
    
    ##--check that train/validation sets are correct size--
    assert(num_patches == len(patches))
    assert(num_train_patches == 0.9*len(patches))
    assert(num_valid_patches == 0.1*len(patches))
    
    ##--get number of batches per epoch--
    num_train_batches = int(0.9*len(patches)//batch_size)
    num_valid_batches = int(0.1*len(patches)//val_batch_size)
    
    ##--write metrics and graph to TensorBoard summary--
    summary = tf.summary.FileWriter(test_path, sess.graph)

    for epoch in range(num_epochs):
        curr_loss = []
        curr_accuracy = []
        
        ##--shuffle data each epoch--
        index_list = [index for index in range(num_patches)]
        shuffle(index_list)
        shuffled_patches = patches[index_list, ...]
        shuffled_ground_truths = patches_gt[index_list, ...]
        
        ##--split data for training/validation--
        train_patches, val_patches, train_gt, val_gt = model_selection.train_test_split(shuffled_patches, shuffled_ground_truths, test_size=0.1)
        
        for i in range(num_train_batches):
            train_batch = train_patches[i*batch_size:min((i+1)*batch_size, num_train_patches)]
            train_batch_gt = train_gt[i*batch_size:min((i+1)*batch_size, num_train_patches)]
            
            if i == 0:
                batch_predictions, batch_loss, _, gn_summ = sess.run([predictions, cost, min_cost, grad_norm_summary], feed_dict={images: train_batch, grnd_truths:train_batch_gt})
                summary.add_summary(gn_summ, epoch)
            
            else:
                batch_predictions, batch_loss, _ = sess.run([predictions, cost, min_cost], feed_dict={images: train_batch, grnd_truths:train_batch_gt})
            
            curr_loss.append(batch_loss)
            curr_accuracy.append(compute_accuracy(batch_predictions, train_batch_gt))
        
        train_loss = np.mean(curr_loss)
        train_accuracy = np.mean(curr_accuracy)
        train_loss_arr.append(train_loss)
        train_accuracy_arr.append(train_accuracy)
        
        curr_loss = []
        curr_accuracy = []
        
        for i in range(num_valid_batches):
            val_batch = val_patches[i*batch_size:min((i+1)*batch_size, num_valid_patches)]
            val_batch_gt = val_gt[i*batch_size:min((i+1)*batch_size, num_valid_patches)]
        
            val_batch_predictions, val_batch_loss = sess.run([predictions, cost], feed_dict={images: val_batch, grnd_truths:val_batch_gt})
            
            curr_loss.append(val_batch_loss)
            curr_accuracy.append(compute_accuracy(val_batch_predictions, val_batch_gt))
        
        val_loss = np.mean(curr_loss)
        val_accuracy = np.mean(curr_accuracy)
        val_loss_arr.append(val_loss)
        val_accuracy_arr.append(val_accuracy)
        
        print("Epoch " + str(epoch+1) + "/" + str(num_epochs) + ", train loss = " + \
              "{:0.6f}".format(train_loss) + ", train accuracy = " + \
              "{:0.5f}".format(train_accuracy))
        
        print("{0: <14}".format('') + "valid loss = " + \
              "{:0.6f}".format(val_loss) + ", valid accuracy = " + \
              "{:0.5f}".format(val_accuracy) + "\n")
        
        if(val_accuracy > best_accuracy):
            save_path = saver.save(sess, model_path)
            print("{0:<6}".format("Accuracy increased from ") + str(best_accuracy) + " to " + str(val_accuracy))
            print("Saving model in %s" % model_path + "\n")
            best_accuracy = val_accuracy
        
        summ = sess.run(performance, feed_dict={loss_ph: val_loss, accuracy_ph: val_accuracy})
        summary.add_summary(summ, epoch)
        
    summary.close()
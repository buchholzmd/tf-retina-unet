import numpy as np
import tensorflow as tf

from random import shuffle
from eval_helper import *
from model import *

#----------------VARIABLES----------------
patch_height = 48
patch_width = 48
channels = 1
num_patches = 190000
inside_FOV = False

num_epochs = 150
learn_rate1 = 0.01
learn_rate2 = 0.001
momentum = 0.3
batch_size = 32
dataset_path = "./datasets/"
summaries_path = "./summaries/"

images = tf.placeholder(tf.float32, shape=(None, patch_height, patch_width, channels), name='Input')
grnd_truths = tf.placeholder(tf.float32, shape=(None, patch_height*patch_width, 2), name='Labels')

#after running the model with xavier initializers try He init (Variance Scaling) for ReLU!!
weights = {
    'w1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w2': tf.get_variable('W1', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w3': tf.get_variable('W2', shape=(3, 3, 32, 16), initializer=tf.contrib.layers.xavier_initializer()),
    'w4': tf.get_variable('W3', shape=(3, 3, 16, 16), initializer=tf.contrib.layers.xavier_initializer()),
    'w5': tf.get_variable('W4', shape=(3, 3, 16, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w6': tf.get_variable('W5', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w7': tf.get_variable('W6', shape=(3, 3, 32, 64), initializer=tf.contrib.layers.xavier_initializer()),
    'w8': tf.get_variable('W7', shape=(3, 3, 64, 64), initializer=tf.contrib.layers.xavier_initializer()),
    'w9': tf.get_variable('W8', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),
    'w10': tf.get_variable('W9', shape=(3, 3, 128, 128), initializer=tf.contrib.layers.xavier_initializer()),
    'w11': tf.get_variable('W10', shape=(3, 3, 192, 64), initializer=tf.contrib.layers.xavier_initializer()),
    'w12': tf.get_variable('W11', shape=(3, 3, 64, 64), initializer=tf.contrib.layers.xavier_initializer()),
    'w13': tf.get_variable('W12', shape=(3, 3, 96, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w14': tf.get_variable('W13', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w15': tf.get_variable('W14', shape=(3, 3, 48, 16), initializer=tf.contrib.layers.xavier_initializer()),
    'w16': tf.get_variable('W15', shape=(3, 3, 16, 16), initializer=tf.contrib.layers.xavier_initializer()),
    'w17': tf.get_variable('W16', shape=(3, 3, 16, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w18': tf.get_variable('W17', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'w19': tf.get_variable('W18', shape=(1, 1, 32, 2), initializer=tf.contrib.layers.xavier_initializer())
}

biases = {
    'b1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b3': tf.get_variable('B2', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
    'b4': tf.get_variable('B3', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
    'b5': tf.get_variable('B4', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b6': tf.get_variable('B5', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b7': tf.get_variable('B6', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'b8': tf.get_variable('B7', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'b9': tf.get_variable('B8', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'b10': tf.get_variable('B9', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'b11': tf.get_variable('B10', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'b12': tf.get_variable('B11', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'b13': tf.get_variable('B12', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b14': tf.get_variable('B13', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b15': tf.get_variable('B14', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
    'b16': tf.get_variable('B15', shape=(16), initializer=tf.contrib.layers.xavier_initializer()),
    'b17': tf.get_variable('B16', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b18': tf.get_variable('B17', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'b19': tf.get_variable('B18', shape=(2), initializer=tf.contrib.layers.xavier_initializer())
}
#-----------------------------------------

## get data
patches, patches_gt = get_data(dataset_path, "train")

##get model, loss and optimizer 
model = unet(images, weights, biases)

predictions = tf.nn.softmax(model, name="predictions")
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=grnd_truths), name="Loss")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate1)
#optimizer = tf.train.MomentumOptimizer(learning_rate=learn_rate1, momentum=momentum).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate2, momentum=momentum).minimize(cost)

#compute gradients and minimize loss function
gradients = optimizer.compute_gradients(cost)
min_cost = optimizer.minimize(cost)

#get summaries
performance, grad_norm_summary, loss_ph, accuracy_ph = visualize_performance(gradients)

#configure GPU memory usage
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

if not os.path.exists(summaries_path):
    os.makedirs(summaries_path)
    
init = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(init)
    train_loss = []
    #val_loss = []
    train_accuracy = []
    #val_accuracy = []
    
    num_batches = len(patches)//batch_size
    summary = tf.summary.FileWriter(summaries_path, sess.graph)

    for epoch in range(num_epochs):
        curr_loss = []
        curr_accuracy = []
        
        #shuffle data each epoch
        index_list = [index for index in range(num_patches)]
        shuffle(index_list)
        shuffled_patches = patches[index_list, ...]
        shuffled_ground_truths = patches_gt[index_list, ...]
        
        for i in range(num_batches):
            batch = shuffled_patches[i*batch_size:min((i+1)*batch_size, num_patches)]
            batch_gt = shuffled_ground_truths[i*batch_size:min((i+1)*batch_size, num_patches)]
            
            if i == 0:
                batch_predictions, batch_loss, _, gn_summ = sess.run([predictions, cost, min_cost, grad_norm_summary], feed_dict={images: batch, grnd_truths:batch_gt})
                summary.add_summary(gn_summ, epoch)
            
            else:
                batch_predictions, batch_loss, _ = sess.run([predictions, cost, min_cost], feed_dict={images: batch, grnd_truths:batch_gt})
            
            curr_loss.append(batch_loss)
            curr_accuracy.append(compute_accuracy(batch_predictions, batch_gt))
        
        loss = np.mean(curr_loss)
        accuracy = np.mean(curr_accuracy)
        
        train_loss.append(loss)
        train_accuracy.append(accuracy)
        
        print("Epoch " + str(epoch+1) + "/" + str(num_epochs) + ", train loss = " + \
              "{:0.6f}".format(loss) + ", train accuracy = " + \
              "{:0.5f}".format(accuracy) + "\n")
            
    summ = sess.run(performance, feed_dict={loss_ph: loss, accuracy_ph: accuracy})
    summary.close()
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, jaccard_similarity_score, f1_score
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import sys
sys.path.insert(0, './model/')
from input import masks_test, get_masks
from random import shuffle
from eval_helper import *
from model import *
from file_processing import *

#----------------VARIABLES----------------
patch_height = 48
patch_width = 48
channels = 1
num_patches = 190000
orig_height = 565
orig_width = 565
orig_channels = 3

group_size = 1
stride = 5
num_test_batches = 125
dataset_path = "./datasets/"
results_path = "./results/"
model_path = "./test/model_2.ckpt"

images = tf.placeholder(tf.float32, shape=(None, patch_height, patch_width, channels), name='Input')
grnd_truths = tf.placeholder(tf.float32, shape=(None, patch_height*patch_width, 2), name='Labels')

weights = {
    'w1': tf.get_variable('W0', shape=(3, 3, 1, 32)),
    'w2': tf.get_variable('W1', shape=(3, 3, 32, 32)),
    'w3': tf.get_variable('W2', shape=(3, 3, 32, 16)),
    'w4': tf.get_variable('W3', shape=(3, 3, 16, 16)),
    'w5': tf.get_variable('W4', shape=(3, 3, 16, 32)),
    'w6': tf.get_variable('W5', shape=(3, 3, 32, 32)),
    'w7': tf.get_variable('W6', shape=(3, 3, 32, 64)),
    'w8': tf.get_variable('W7', shape=(3, 3, 64, 64)),
    'w9': tf.get_variable('W8', shape=(3, 3, 64, 128)),
    'w10': tf.get_variable('W9', shape=(3, 3, 128, 128)),
    'w11': tf.get_variable('W10', shape=(3, 3, 192, 64)),
    'w12': tf.get_variable('W11', shape=(3, 3, 64, 64)),
    'w13': tf.get_variable('W12', shape=(3, 3, 96, 32)),
    'w14': tf.get_variable('W13', shape=(3, 3, 32, 32)),
    'w15': tf.get_variable('W14', shape=(3, 3, 48, 16)),
    'w16': tf.get_variable('W15', shape=(3, 3, 16, 16)),
    'w17': tf.get_variable('W16', shape=(3, 3, 16, 32)),
    'w18': tf.get_variable('W17', shape=(3, 3, 32, 32)),
    'w19': tf.get_variable('W18', shape=(1, 1, 32, 2))
}

biases = {
    'b1': tf.get_variable('B0', shape=(32)),
    'b2': tf.get_variable('B1', shape=(32)),
    'b3': tf.get_variable('B2', shape=(16)),
    'b4': tf.get_variable('B3', shape=(16)),
    'b5': tf.get_variable('B4', shape=(32)),
    'b6': tf.get_variable('B5', shape=(32)),
    'b7': tf.get_variable('B6', shape=(64)),
    'b8': tf.get_variable('B7', shape=(64)),
    'b9': tf.get_variable('B8', shape=(128)),
    'b10': tf.get_variable('B9', shape=(128)),
    'b11': tf.get_variable('B10', shape=(64)),
    'b12': tf.get_variable('B11', shape=(64)),
    'b13': tf.get_variable('B12', shape=(32)),
    'b14': tf.get_variable('B13', shape=(32)),
    'b15': tf.get_variable('B14', shape=(16)),
    'b16': tf.get_variable('B15', shape=(16)),
    'b17': tf.get_variable('B16', shape=(32)),
    'b18': tf.get_variable('B17', shape=(32)),
    'b19': tf.get_variable('B18', shape=(2))
}
#-----------------------------------------

new_size, patches, test_gts = get_data(dataset_path + "test/", "test")

if not os.path.exists(dataset_path + "test/predictions.hdf5"):
    ##--get model, and logits--
    model = unet(images, weights, biases)
    predict = tf.nn.softmax(model, name="predictions")
    
    saver = tf.train.Saver()
    
    predictions = np.empty((0, predict.shape[1], predict.shape[2]))
    
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        print("--Restored model with best weights--")
    
        images_per_batch = patches.shape[0]//num_test_batches
    
        for i in range(num_test_batches):
            test_batch = patches[i*images_per_batch:(i+1)*images_per_batch,...]
            batch_pred = sess.run(predict, feed_dict={images: test_batch})
            predictions = np.concatenate((predictions, batch_pred), axis=0)
            print(str(i+1) + " passed!")
        
    write(predictions, dataset_path + "test/predictions.hdf5")

else:
    predictions = load(dataset_path + "test/predictions.hdf5")
#--convert model predictions to images--
pred_patches = conv_to_imgs(predictions, patch_height, patch_width, stride)

#--visualize the predictions--
pred_images = recombine_patches(pred_patches, new_size, stride)
masks = get_masks(masks_test, "test")

#--remove the border on predictions--
remove_border(pred_images, masks)

#--return to original dimensions--
pred_images = pred_images[:, 0:orig_height, 0:orig_width, :]
test_gts = test_gts[:, 0:orig_height, 0:orig_width, :]

if not os.path.exists(results_path):
    os.makedirs(results_path)

#--save columns of predictions--
save_image(group(pred_images), results_path + "predictions.png")
save_image(group(test_gts), results_path + "ground_truths.png")

#--save individual predictions--
num_predictions = pred_images.shape[0]

for i in range(num_predictions):
    img_strip = np.concatenate((pred_images[i,...], test_gts[i,...]), axis=0)
    save_image(img_strip, results_path + "test_prediction" + str(i+1) + ".png")

#--Analytics--
pred_scores, ground_truths = prediction_in_FOV(pred_images, test_gts, masks)

roc_auc = roc_auc_score(ground_truths, pred_scores)
false_positives, true_positives, thresholds = roc_curve(ground_truths, pred_scores)

roc_curve = plt.figure()
plt.plot(false_positives, true_positives, '-', label='Area under curve = %0.4f' % roc_auc)
plt.title('ROC Curve')
plt.xlabel('False positive rate')
plt.xlabel('True positive rate')
plt.legend(loc="lower right")
plt.savefig(results_path + "roc_curve.png")

#--Confusion matrix--
preds = np.empty((pred_scores.shape[0]))
for i in range(pred_scores.shape[0]):
    if pred_scores[i] >= 0.5:
        preds[i] = 1
    else:
        preds[i] = 0

con_mat = confusion_matrix(ground_truths, preds)

##
#Confusion Matrix Anlaytics
##

#--Jaccard Similarity--
jaccard_sim = jaccard_similarity_score(ground_truths, preds, normalize=True)

#--F1 score--
f1 = f1_score(ground_truths, preds, labels=None, average='binary', sample_weight=None)

print("--------Printing analytics--------")
print("Area under ROC curve: " + str(roc_auc) + "\n")
print(" -------Confusion Matrix-------")
print(con_mat)
print("Jaccard similarity: " + str(jaccard_sim) + "\n")
print("F1 score: " + str(f1))
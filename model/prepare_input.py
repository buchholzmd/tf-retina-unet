import random

from input import *
from pre_processing import *

def extract_patches(patch_height, patch_width, num_patches, inside=True):
    '''
        This function extracts the sub-images of the pre-processd full images 

        Args:
          patch_height: int, height of patches
          patch_width: int, width of patches
          num_patches: int, number of patches to extract
          inside: bool, flag to indicate whether training or test data
          
        Returns:
          patches: numpy.array, 4D array of random training patches
          patches_gt: numpy.array, 4D array of random ground truths of training patches
    '''
    orig_train_imgs, train_grnd_truths, train_masks = get_data(orig_imgs_train, grnd_truths_train, masks_train, dataset="train")

    #reshape dimensions of ground truths and border masks
    train_grnd_truths = np.reshape(train_grnd_truths, (num_imgs, height, width, 1))
    train_masks = np.reshape(train_masks, (num_imgs, height, width, 1))

    #augment data and normalize ground truth pixel data
    train_imgs = augment_data(orig_train_imgs)
    train_grnd_truths = train_grnd_truths / 255

    #cut bottom and top data is 565x565
    train_imgs = train_imgs[:,9:574,:,:]
    train_grnd_truths = train_grnd_truths[:,9:574,:,:]

    #extract patches
    patches = np.empty((num_patches, patch_height, patch_width, train_imgs.shape[3]))
    patches_gt = np.empty((num_patches, patch_height, patch_width, train_grnd_truths.shape[3]))
    img_height = train_imgs.shape[1]
    img_width = train_imgs.shape[2]

    #get equal number of patches per image
    patch_per_img = int(num_patches/train_imgs.shape[0])

    count = 0
    for i in range(len(train_imgs)):
        j = 0
        while j < patch_per_img:
            #get random x,y center for patch
            x = random.randint(int(patch_width/2),img_width-int(patch_width/2))
            y = random.randint(int(patch_height/2), img_height-int(patch_height/2))

            #check that patch is fully within FOV
            if inside == True:
                if not inside_FOV(x, y, img_height, img_width, patch_height):
                    continue
            patch_bottom = y - int(patch_height/2)
            patch_top = y + int(patch_height/2)
            patch_left = x - int(patch_width/2)
            patch_right = x + int(patch_width/2)
                
            patch = train_imgs[i, patch_bottom:patch_top, patch_left:patch_right, :]
            patch_gt = train_grnd_truths[i, patch_bottom:patch_top, patch_left:patch_right, :]

            patches[count] = patch
            patches_gt[count] = patch_gt
            count += 1
            j += 1

    return patches, patches_gt

def inside_FOV(x, y, height, width, patch):
    x_ = x - int(width/2)
    y_ = y - int(height/2)

    #radius is 270 from DRIVE documentation
    inside_rad = 270 - int(patch / np.sqrt(2))
    radius = np.sqrt(x_**2 + y_**2)

    return radius < inside_rad

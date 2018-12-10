import random

from input import *
from pre_processing import *

def extract_train_patches(patch_height, patch_width, num_patches, inside=True):
    '''
        This function extracts the sub-images of the pre-processd full images

        Args:
          patch_height: int, height of patches
          patch_width: int, width of patches
          num_patches: int, number of patches to extract
          inside: bool, flag to indicate whether should consider pixels only inside field of view
          
        Returns:
          patches: numpy.array, 4D array of random training patches
          patches_gt: numpy.array, 4D array of random ground truths of training patches
    '''
    orig_train_imgs, train_grnd_truths = get_dataset(orig_imgs_train, grnd_truths_train)

    #reshape dimensions of ground truths
    train_grnd_truths = np.reshape(train_grnd_truths, (num_imgs, height, width, 1))

    #augment data and normalize ground truth pixel data
    print("---------------------------------------")
    print("Augmenting data")
    print("---------------------------------------\n")
    train_imgs = augment_data(orig_train_imgs)
    train_grnd_truths = train_grnd_truths / 255
    print("\tData augmentation complete\n")

    #cut bottom and top data is 565x565
    train_imgs = train_imgs[:,9:574,:,:]
    train_grnd_truths = train_grnd_truths[:,9:574,:,:]

    #extract patches
    print("---------------------------------------")
    print("Extracting training patches")
    print("---------------------------------------\n")
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

    print("\tPatch extraction complete\n")
    return patches, patches_gt

def extract_test_patches(patch_height, patch_width, stride):
    '''
        This function extracts the sub-images of the pre-processd full images for testing

        Args:
          patch_height: int, height of patches
          patch_width: int, width of patches
          stride: int, size of stride for window during patch extraction
          
        Returns:
          new_size: tuple, new dimensions of images, after padding is added, and before patch extraction (used to recombine patches into the image prediction)
          patches: numpy.array, 4D array of extracted test patches
          test_grnd_truths: numpy.array, 4D array of ground truths of of test images
    '''
    orig_test_imgs, test_grnd_truths = get_dataset(orig_imgs_test, grnd_truths_test)
    
    #reshape dimensions of ground truths
    test_grnd_truths = np.reshape(test_grnd_truths, (num_imgs, height, width, 1))
    
    #augment data and normalize ground truth pixel data
    print("---------------------------------------")
    print("Augmenting data")
    print("---------------------------------------\n")
    test_imgs = augment_data(orig_test_imgs)
    test_grnd_truths = test_grnd_truths / 255
    print("\tData augmentation complete\n")

    #cut bottom and top data is 565x565
    test_imgs = test_imgs[:,9:574,:,:]
    test_grnd_truths = test_grnd_truths[:,9:574,:,:]
    
    #check ground truths are between 0-1
    assert(np.max(test_grnd_truths)==1  and np.min(test_grnd_truths)==0)
    
    #paint border overlap
    print("---------------------------------------")
    print("Painting border overlap")
    print("---------------------------------------\n")
    test_imgs = border_overlap(test_imgs, patch_height, patch_width, stride)
    new_size = (test_imgs.shape[1], test_imgs.shape[2])
    
    new_num_imgs = test_imgs.shape[0]
    new_height = test_imgs.shape[1]
    new_width = test_imgs.shape[2]
    new_channels = test_imgs.shape[3]
    
    #check padding yeilds correct dimensions
    assert((new_height-patch_height)%stride==0 and (new_width-patch_width)%stride==0)
    
    patches_per_img = ((new_height-patch_height)//stride+1)*((new_width-patch_width)//stride+1)
    num_patches = patches_per_img*new_num_imgs
    
    patches = np.empty((num_patches, patch_height, patch_width, new_channels))
    count = 0
    
    #extract patches
    print("---------------------------------------")
    print("Extracting testing patches")
    print("---------------------------------------\n")
    for i in range(new_num_imgs):
        for j in range((new_height-patch_height)//stride+1):
            for k in range((new_width-patch_width)//stride+1):
                patch = test_imgs[i, j*stride:(j*stride)+patch_height, k*stride:(k*stride)+patch_width]
                patches[count] = patch
                count += 1
    
    assert(count==num_patches)
    return new_size, patches, test_grnd_truths
    
    
    
def inside_FOV(x, y, height, width, patch):
    '''
        Function to check whether a given pixel coordinate (x, y) is within the field of view

        Args:
          x: int, x dimension for given pixel
          y: int, y dimension for given pixel
          height: int, full image height
          width: int, full image width
          patch: int, height/width of a patch
          
        Returns:
          new_gts: numpy.array, 3D array of ground truth patches (with correct shape)
    '''
    x_ = x - int(width/2)
    y_ = y - int(height/2)

    #radius is 270 from DRIVE documentation
    inside_rad = 270 - int(patch / np.sqrt(2))
    radius = np.sqrt(x_**2 + y_**2)

    return radius < inside_rad

def prepare_grnd_truths(grnd_truths):
    '''
        Function to prepare the ground-truth patches for evaluation of the model during training 

        Args:
          grnd_truths: numpy.array, 4D array of ground truth patches
          
        Returns:
          new_gts: numpy.array, 3D array of ground truth patches (with correct shape)
    '''
    assert(len(grnd_truths.shape) == 4)
    assert(grnd_truths.shape[3] == 1)
    print("---------------------------------------")
    print("Preparing ground truths for training")
    print("---------------------------------------\n")
    batches = grnd_truths.shape[0]
    height = grnd_truths.shape[1]
    width = grnd_truths.shape[2]
    
    grnd_truths = np.reshape(grnd_truths, (batches, height*width))
    new_gts = np.empty((batches, height*width, 2))
    
    for i in range(batches):
        for j in range(height*width):
            if grnd_truths[i, j] == 0:
                new_gts[i, j, 0] = 1
                new_gts[i, j, 1] = 0
            else:
                new_gts[i, j, 0] = 0
                new_gts[i, j, 1] = 1
                
    print("\tGround truth preparation complete\n")
    return new_gts

def border_overlap(imgs, patch_height, patch_width, stride):
    '''
        Function to add padding to guarantee image dimensions ascribe to stride size

        Args:
          imgs: numpy.array, images to be padded for uniform patch extraction
          patch_height: int, height of patches
          patch_width: int, width of patches
          stride: int, stride from window when extracting patches
          
        Returns:
          imgs: numpy.array, 4D nunmpy array containing padded images ready for patch extraction
    '''
    assert(len(imgs.shape)==4)
    assert(imgs.shape[3]==1)
    height = imgs.shape[1]
    width = imgs.shape[2]
    channels = imgs.shape[3]
    
    extra_h = (height - patch_height) % stride
    extra_w = (width - patch_width) % stride
    
    if(extra_h != 0):
        print("---------------------------------------")
        print("Padding height with extra pixels")
        print("---------------------------------------\n")
        
        tmp_imgs = np.zeros((imgs.shape[0], height+(stride-extra_h), width, channels))
        tmp_imgs[0:imgs.shape[0], 0:height, 0:width, 0:channels] = imgs
        imgs = tmp_imgs
    
    #--in case dimensions have changed--
    height = imgs.shape[1]
    width = imgs.shape[2]
    channels = imgs.shape[3]
        
    if(extra_w != 0):
        print("---------------------------------------")
        print("Padding width with extra pixels")
        print("---------------------------------------\n")
        
        tmp_imgs = np.zeros((imgs.shape[0], height, width+(stride-extra_w), channels))
        tmp_imgs[0:imgs.shape[0], 0:height, 0:width, 0:channels] = imgs
        imgs = tmp_imgs
    
    return imgs
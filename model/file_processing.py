import h5py
import numpy as np
import _pickle as cPickle
from PIL import Image

def write(data, outfile):
    '''
        This function writes the pre-processed image data to a HDF5 file

        Args:
          data: numpy.array, image data as numpy array
          outfile: string, path to write file to
    '''
    print("---------------------------------------")
    print("Saving data")
    print("---------------------------------------\n")
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=data, dtype=data.dtype)

def load(infile):
    '''
        This function loads the image data from a HDF5 file 

        Args:
          outfile: string, path to read file from
          
        Returns:
          f["image"][()]: numpy.array, image data as numpy array
    '''
    print("---------------------------------------")
    print("Loading data")
    print("---------------------------------------\n")
    with h5py.File(infile, "r") as f:
        return f["image"][()]
    
def write_tuple(data, outfile):
    with open(outfile, 'wb') as f:
        cPickle.dump(data, f)
    
def load_tuple(infile):
    with open(infile, 'rb') as f:
        return cPickle.load(f)

def save_image(data, path):
    assert(len(data.shape)==3)
    
    print("---------------------------------------")
    print("Saving images")
    print("---------------------------------------\n")
    #--check if data is greyscale
    if data.shape[2] == 1:
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    #--check if data is 0-255--
    if np.max(data) > 1.0:
        image = Image.fromarray(data.astype(np.uint8))
    else:
        image = Image.fromarray((data*255).astype(np.uint8))
    
    image.save(path)
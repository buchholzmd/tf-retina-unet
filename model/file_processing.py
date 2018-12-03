import h5py
import numpy as np

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
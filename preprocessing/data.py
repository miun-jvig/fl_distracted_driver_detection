'''
This script converts the (modified) statefarm dataset into a h5 file.
'''

import h5py
import cv2
from pathlib import Path
import glob
import os
import numpy as np
import argparse
import random

def dir_path(string):
    try:
        os.path.isdir(string)
        return string
    except OSError:
        raise (string, 'not a directory')


def load_image(path, rows=256, cols=256):
    img = cv2.imread(path)
    img = cv2.resize(img, (rows, cols), cv2.INTER_LINEAR)
#    resized = img.astype('float32')
#    mean = [103.939, 116.779, 123.68]
#    resized[:, :, 0] -= mean[0]
#    resized[:, :, 1] -= mean[1]
#    resized[:, :, 2] -= mean[2]
    #cv2.imshow("image", img)
    #cv2.imshow("image_orig", cv2.imread(path))
    #cv2.waitKey(0)
    return img #resized.astype('uint8')


def load_data(datapath, subset='train', rows=256, cols=256):
    paths = glob.glob(os.path.join(datapath, subset, "*", "*.jpg"))
    print(len(paths), 'files in total')
    labels = [int(x.split('\\')[-2][1]) for x in paths]
    images = [load_image(x, rows, cols) for x in paths]
    c = list(zip(images, labels))
    random.shuffle(c)
    images, labels = zip(*c)
    return images, labels


def read_hdf5(hdf5_dir=Path('./'), subset='train', rows=256, cols=256):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        hdf5_dir   path where the h5 file is
        rows       number of pixel rows (in the image)
        cols        number of pixel cols (in the image)
        Returns:
        ----------
        images      images array, (N, rows, cols, 3) to be read
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"driver_distraction_{rows}x{cols}_{subset}.h5", "r+")
    #images = np.array(file["/images"]).astype(float)
    #labels = np.array(file["/meta"]).astype(int)
    return file #images, labels


def store_hdf5(images, labels, out_dir=Path('./'), subset='train', rows=256, cols=256):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images      images array, (N, rows, cols, 3) to be stored
        labels      labels array, (N, 1) to be stored
        out_dir     path where to store the h5 file
        rows        number of rows per image
        cols        number of cols per image
    """
    # Create a new HDF5 file
    file = h5py.File(out_dir / f"driver_distraction_{rows}x{cols}_{subset}.h5", "w")
    # Create a dataset in the file
    print('\tas ',file)
    dataset = file.create_dataset(
        "images", np.shape(images), np.uint8, data=images, compression="gzip", chunks=True
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), np.uint8, data=labels
    )
    file.close()


def convert(datapath, output, subset, rows, cols):
    hdf5_dir = Path(output)
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    print('Reading the dataset')
    images, labels = load_data(datapath, subset, rows, cols)
    print('Writing the dataset')
    store_hdf5(images, labels, hdf5_dir, subset, rows, cols)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Converting the whole dataset into one h5 file.')
    parser.add_argument("-d", "--datapath", help="path of the dataset", type=dir_path, required=True)
    parser.add_argument("-o", "--destpath", help="path where to write the output file",required=True)
    parser.add_argument("-r", "--rows", help="row size, default 256",type=int, default=256)
    parser.add_argument("-c", "--cols", help="column size, default 256",type=int, default=256)
    parser.add_argument("-s", "--set", help="data subset could be train or test", type=str, default='train')
    args = parser.parse_args()
    print(args)
    convert(args.datapath, args.destpath, args.set, args.rows, args.cols)

# python preprocessing\data.py -o h5 -d data/statefarm/ -r 128 -c 128 -s train
# python preprocessing\data.py -o h5 -d data/statefarm/ -r 128 -c 128 -s test

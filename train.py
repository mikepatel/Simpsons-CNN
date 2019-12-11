"""
Michael Patel
December 2019

Python 3.6.5
TensorFlow 2.0.0

Project description:
    Classifier for characters from The Simpsons

File description:
    Preprocessing and training algorithm

"""
################################################################################
# Imports
import os
import numpy as np
from datetime import datetime
import glob  # module to find all pathnames matching a specified pattern
import imageio
from PIL import Image
import cv2

import tensorflow as tf

from parameters import *


################################################################################
# Main
if __name__ == "__main__":
    # print out TF version
    print(f'TF version: {tf.__version__}')

    # eager execution is enabled by default in TF 2.0
    print(f'Using eager execution: {tf.executing_eagerly()}')

    # create output directory for checkpoints, results, images
    output_dir = "results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    images_train = []
    images_test = []
    labels_train = []
    labels_test = []

    # mapping between Simpsons character and a numeric categorial label
    char2num = {
        "Bart Simpson": 0,
        "Homer Simpson": 1
    }

    # reverse mapping
    num2char = {v: k for k, v in char2num.items()}

    # number of output classes
    num_classes = len(char2num)

    for character in char2num:
        image_files_pattern = os.path.join(os.getcwd(), "data\\" + character) + "\\*.jpg"
        filenames = glob.glob(image_files_pattern)

        # create temporary (images, labels) dataset, and then split into training and test
        temp_images = []
        temp_labels = []

        for f in filenames:
            # imageio: RGB
            #image = imageio.imread(f)
            #Image.fromarray(image).show()

            # cv2: BGR
            #x = cv2.imread(f)  # inverts RGB colour scheme to BGR
            #Image.fromarray(x).show()

            # PIL Image: RGB
            image = Image.open(f)
            #print(image)
            #image = np.array(image)
            #print(image)
            #Image.fromarray(image).show()

            # resize image
            resized_image = image.resize((64, 64))  # resize() returns a new image object
            #print(x)
            #x.show()
            #image = np.array(image)
            #Image.fromarray(image).show()

            # (images as arrays, numeric categorical labels)
            temp_images.append(np.array(resized_image))
            temp_labels.append(char2num[character])

        # create temp training and test sets for each Simpsons character
        split_idx = int(DATA_SPLIT_PERCENTAGE * len(temp_images))

        # split images
        temp_images_train = temp_images[:split_idx]
        temp_images_test = temp_images[split_idx:]

        # split labels
        temp_labels_train = temp_labels[:split_idx]
        temp_labels_test = temp_labels[split_idx:]

        # join temp training and test sets for each Simpsons character together
        images_train = images_train + temp_images_train
        images_test = images_test + temp_images_test

        labels_train = labels_train + temp_labels_train
        labels_test = labels_test + temp_labels_test

    # normalizing data
    images_train = np.array(images_train).astype("float32") / 255.
    images_test = np.array(images_test).astype("float32") / 255.

    labels_train = np.array(labels_train).astype("float32") / 255.
    labels_test = np.array(labels_test).astype("float32") / 255.

    # verify shape of data
    print(f'Training images shape: {images_train.shape}')
    print(f'Training labels shape: {labels_train.shape}')
    print(f'Test images shape: {images_test.shape}')
    print(f'Test labels shape: {labels_test.shape}')

    input_shape = (64, 64, 3)

    # ----- MODEL ----- #

    quit()


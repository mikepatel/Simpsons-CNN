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
    images = []
    labels = []

    # mapping between Simpsons character and a numeric categorial label
    char2num = {
        "Bart Simpson": 0,
        "Homer Simpson": 1
    }

    for character in char2num:
        image_files_pattern = os.path.join(os.getcwd(), "data\\" + character) + "\\*.jpg"
        filenames = glob.glob(image_files_pattern)
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

            images.append(resized_image)
            labels.append(char2num[character])

    print(len(images))
    print(len(labels))

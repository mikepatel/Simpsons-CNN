"""
Michael Patel
April 2021

Project description:
    Use TensorFlow to create a Simpsons character classifier that runs on an Android device

File description:
    For imports and model/training hyperparameters

"""
################################################################################
# Imports
import os
import matplotlib.pyplot as plt
import tensorflow as tf


################################################################################
# directories
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MOBILENET_DIR = os.path.join(BASE_DIR, "mobilenetv2")
SAVE_DIR = os.path.join(MOBILENET_DIR, "saved")

# image dimensions
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

# model and training parameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-3  # Adam

# fine-tuning
NUM_LAYERS_FREEZE = 100
NUM_EPOCHS_FINE_TUNING = 1
LEARNING_RATE_FINE_TUNING = 1e-2  # SGD

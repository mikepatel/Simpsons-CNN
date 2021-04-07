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
CUSTOM_DIR = os.path.join(BASE_DIR, "custom")
SAVE_DIR = os.path.join(CUSTOM_DIR, "saved")

# image dimensions
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

# model and training parameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-3  # Adam

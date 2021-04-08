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
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
import glob
import imageio
from PIL import Image, ImageDraw, ImageFont


################################################################################
# directories
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAVE_DIR = os.path.join(BASE_DIR, "saved")
PREDICTIONS_DIR = os.path.join(SAVE_DIR, "predictions")

# image dimensions
IMAGE_WIDTH = 224  # 64
IMAGE_HEIGHT = 224  # 64
IMAGE_CHANNELS = 3

# model and training parameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-3  # Adam

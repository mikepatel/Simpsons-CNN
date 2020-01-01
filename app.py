"""
Michael Patel
December 2019

Python 3.6.5
TensorFlow 2.0.0

Project description:
    Classifier for characters from The Simpsons

File description:
    Application run for model once model is trained

"""
################################################################################
# Imports
import os
import tensorflow as tf


################################################################################
# Main
if __name__ == "__main__":
    # ----- PREDICT ----- #
    # load model
    filepath = os.path.join(os.getcwd(), "temp\\saved_model.h5")
    model = tf.keras.models.load_model(filepath)

    # predict on new data

    # create gif

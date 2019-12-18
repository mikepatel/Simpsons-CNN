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
import tensorflow as tf


################################################################################
# Main
if __name__ == "__main__":
    # ----- PREDICT ----- #
    # load model
    model = tf.keras.models.load_model(output_dir + "\\saved_model.h5")

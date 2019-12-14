"""
Michael Patel
December 2019

Python 3.6.5
TensorFlow 2.0.0

Project description:
    Classifier for characters from The Simpsons

File description:
    Model definitions

"""
################################################################################
# Imports
import tensorflow as tf


################################################################################
# CNN
def build_cnn(input_shape, num_classes):
    m = tf.keras.Sequential()

    # ----- Stage 1 ----- #
    # Convulation
    m.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[3, 3],
        input_shape=input_shape,
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Convolution
    m.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Max Pooling
    m.add(tf.keras.layers.MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    # Dropout
    m.add(tf.keras.layers.Dropout(
        rate=0.25
    ))

    # ----- Stage 2 ----- #
    # Convulation
    m.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Convolution
    m.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Max Pooling
    m.add(tf.keras.layers.MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    # Dropout
    m.add(tf.keras.layers.Dropout(
        rate=0.25
    ))

    # ----- Stage 3 ----- #
    # Convolution
    m.add(tf.keras.layers.Conv2D(
        units=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Convolution
    m.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.keras.activations.relu
    ))

    # Max Pooling
    m.add(tf.keras.layers.MaxPool2D(
        pool_size=[2, 2],
        strides=2
    ))

    # Dropout
    m.add(tf.keras.layers.Dropout(
        rate=0.25
    ))

    # ----- Stage 4 ----- #
    # Flatten
    m.add(tf.keras.layers.Flatten())

    # Dense
    m.add(tf.keras.layers.Dense(
        units=1024,
        activation=tf.keras.activations.relu
    ))

    # Dropout
    m.add(tf.keras.layers.Dropout(
        rate=0.5
    ))

    # Dense - output
    m.add(tf.keras.layers.Dense(
        units=num_classes,
        activation=tf.keras.activations.softmax
    ))

    return m

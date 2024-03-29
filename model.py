"""
Michael Patel
April 2021

Project description:
    Use TensorFlow to create a Simpsons character classifier that runs on an Android device

File description:
    For model definitions

"""
################################################################################
# Imports
from packages import *


################################################################################
# custom CNN
def build_model(num_classes):
    def block(t, filters):
        t = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            activation=tf.keras.activations.relu
        )(t)
        t = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            activation=tf.keras.activations.relu
        )(t)
        t = tf.keras.layers.MaxPool2D(
            pool_size=2
        )(t)
        t = tf.keras.layers.Dropout(rate=0.3)(t)
        return t

    inputs = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    x = inputs
    x = block(x, filters=32)
    x = block(x, filters=64)
    x = block(x, filters=128)
    x = block(x, filters=256)
    x = block(x, filters=512)  # 224x224x3 --> 7x7x512
    x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=1024, activation=tf.keras.activations.relu)(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)(x)
    outputs = x

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )
    return model

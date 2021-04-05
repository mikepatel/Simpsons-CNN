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
# MobilenetV2
def build_model(num_classes):
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        weights="imagenet",
        include_top=False
    )
    mobilenet.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    x = inputs
    x = mobilenet(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)
    outputs = x

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model

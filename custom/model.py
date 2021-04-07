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
def build_model(num_classes):
    """
    vgg16 = tf.keras.applications.vgg16.VGG16(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        include_top=False
    )

    for layer in vgg16.layers:
        layer.trainable = False

    model = tf.keras.Sequential()
    model.add(vgg16)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation="relu"))
    model.add(tf.keras.layers.Dense(units=num_classes, activation="softmax"))

    return model
    """

    """
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        padding="same",
        activation="relu"
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=2
    ))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=2
    ))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=2
    ))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        padding="same",
        activation="relu"
    ))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=2
    ))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation="relu"))
    #model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=num_classes, activation="softmax"))

    return model
    """

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
        t = tf.keras.layers.Dropout(rate=0.2)(t)
        return t

    inputs = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    x = inputs
    x = block(x, filters=32)
    x = block(x, filters=64)
    x = block(x, filters=128)
    x = block(x, filters=256)
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

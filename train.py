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
import matplotlib.pyplot as plt

import tensorflow as tf

from parameters import *
from model import build_cnn


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
    images_train = []
    images_test = []
    labels_train = []
    labels_test = []

    # mapping between Simpsons character and a numeric categorial label
    char2num = {
        "Bart Simpson": 0,
        "Homer Simpson": 1
    }

    # reverse mapping
    num2char = {v: k for k, v in char2num.items()}

    # number of output classes
    num_classes = len(char2num)
    print(f'Number of classes: {num_classes}')

    for character in char2num:
        image_files_pattern = os.path.join(os.getcwd(), "data\\" + character) + "\\*.jpg"
        filenames = glob.glob(image_files_pattern)

        # create temporary (images, labels) dataset, and then split into training and test
        temp_images = []
        temp_labels = []

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

            # (images as arrays, numeric categorical labels)
            temp_images.append(np.array(resized_image))
            temp_labels.append(char2num[character])

        # create temp training and test sets for each Simpsons character
        split_idx = int(DATA_SPLIT_PERCENTAGE * len(temp_images))

        # split images
        temp_images_train = temp_images[:split_idx]
        temp_images_test = temp_images[split_idx:]

        # split labels
        temp_labels_train = temp_labels[:split_idx]
        temp_labels_test = temp_labels[split_idx:]

        # join temp training and test sets for each Simpsons character together
        images_train = images_train + temp_images_train
        images_test = images_test + temp_images_test

        labels_train = labels_train + temp_labels_train
        labels_test = labels_test + temp_labels_test

    # normalize images
    images_train = np.array(images_train).astype(np.float32) / 255.0
    images_test = np.array(images_test).astype(np.float32) / 255.0

    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)

    # verify shape of data
    print(f'Training images shape: {images_train.shape}')
    print(f'Training labels shape: {labels_train.shape}')
    print(f'Test images shape: {images_test.shape}')
    print(f'Test labels shape: {labels_test.shape}')

    input_shape = (64, 64, 3)

    """
    # convert from array to image
    x = images_train[2000]
    x = x * 255
    x = x.astype(np.uint8)
    Image.fromarray(x).show()
    quit()
    """

    # ----- MODEL ----- #
    # train model
    model = build_cnn(input_shape=input_shape, num_classes=num_classes)
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    model.summary()

    """
    # callbacks --> TensorBoard, save weights
    history_file = output_dir + "\\cnn_train.hdf5"
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=history_file,
        monitor="val_acc",
        save_weights_only=True,
        save_freq=CHECKPOINT_PERIOD
    )
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=output_dir)
    """

    # train model
    history = model.fit(
        x=images_train,
        y=labels_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=images_train.shape[0] // BATCH_SIZE,
        validation_data=(images_test, labels_test)
        #callbacks=[save_callback, tb_callback]
    )

    # plot accuracy
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1.1])
    plt.grid()
    plt.legend(loc="lower right")
    plt.show()

    # save model weights
    #model.save_weights(output_dir + "\\last_checkpoint")

    # load model weights
    #model.load_weights(output_dir + "\\last_checkpoint")

    # evaluate model
    #test_loss, test_accuracy = model.evaluate(images_test, labels_test)
    #print(f'Test accuracy: {test_accuracy:.4f}

    # predictions
    print(labels_test[500])  # ground truth
    predictions = model.predict(images_test)  # predict on images
    x = predictions[500]
    print(x)  # class label as distribution
    print(np.argmax(x))  # class label as int
    print(num2char(np.argmax(x)))  # class label as text

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
from PIL import ImageDraw
from PIL import ImageFont
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
    # Training: 80%
    # Validation: 10%
    # Test: 10%
    images_train = []
    images_test = []
    images_val = []

    labels_train = []
    labels_test = []
    labels_val = []

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

    # create validation set
    midpoint = int(len(images_test) / 2)

    images_val = images_test[:midpoint]
    images_test = images_test[midpoint:]

    labels_val = labels_test[:midpoint]
    labels_test = labels_test[midpoint:]

    # verify shape of data
    print(f'Training images shape: {images_train.shape}')
    print(f'Training labels shape: {labels_train.shape}')
    print(f'Test images shape: {images_test.shape}')
    print(f'Test labels shape: {labels_test.shape}')
    print(f'Validation images shape: {images_val.shape}')
    print(f'Validation labels shape: {labels_val.shape}')

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
        validation_data=(images_val, labels_val)
        #callbacks=[save_callback, tb_callback]
    )

    # plot accuracy
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1.1])
    plt.grid()
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(output_dir + "\\Training Accuracy")

    # evaluate model accuracy to determine overfitting
    test_loss, test_accuracy = model.evaluate(
        x=images_test,
        y=labels_test,
        verbose=0
    )
    print(f'\nTest accuracy: {test_accuracy:.4f}')

    # save the entire model (for later use in Android app)
    model.save(output_dir + "\\saved_model.h5")

    # ----- PREDICT ----- #
    # predictions
    #r = np.random.randint(len(labels_test))
    r = 553  # Homer 2080
    """
    # convert from array to image
    x = images_test[r]
    x = x * 255
    x = x.astype(np.uint8)
    Image.fromarray(x).show()
    """

    print()
    print(f'Test label {r} as int: {labels_test[r]}')  # ground truth as int
    print(f'Test label {r} as text: {num2char[labels_test[r]]}')  # ground truth as text

    i = images_test[r]  # single image
    i = np.expand_dims(i, 0)  # reshape: (1, 64, 64, 3)
    #print(i.shape)

    prediction = model.predict(i)  # predict on image

    # prediction as distribution
    prediction_distribution = prediction
    print(f'Prediction as distribution: {prediction_distribution}')

    # prediction as int
    prediction = np.argmax(prediction_distribution)
    print(f'Prediction as numerical category: {prediction}')

    # prediction as text
    prediction = num2char[prediction]
    print(f'Prediction as text: {num2char[np.argmax(prediction)]}')

    # create prediction text
    t = []
    for i in range(2):
        x = f'{num2char[i]}: {prediction_distribution[0][i]:.6f}'
        t.append(x)

    t = "\n".join(t)
    print(t)

    # write prediction text over image
    image_path = os.path.join(os.getcwd(), "data\\Homer Simpson\\pic_2080.jpg")
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 30)
    draw.text((0, 0), t, font=font)
    image.save(output_dir + "\\pred_image.png")


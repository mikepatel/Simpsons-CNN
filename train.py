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
    print(f'\n# ----- PREPROCESSING ----- #')

    # Training: 80%
    # Validation: 10%
    # Test: 10%
    train_images = []
    val_images = []
    test_images = []

    train_labels = []
    val_labels = []
    test_labels = []

    # mapping between Simpsons character and a numeric categorical label
    char2num = {
        "Bart Simpson": 0,
        "Homer Simpson": 1
    }

    # reverse mapping
    num2char = {v: k for k, v in char2num.items()}

    # number of output classes
    num_classes = len(char2num)
    print(f'Number of classes: {num_classes}')

    # build train, validation, test datasets
    for character in char2num:
        image_files_pattern = os.path.join(os.getcwd(), "data\\Training\\" + character) + "\\*.jpg"
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
        split_idx = int(0.80 * len(temp_images))

        # split images
        temp_train_images = temp_images[:split_idx]
        temp_test_images = temp_images[split_idx:]

        # split labels
        temp_train_labels = temp_labels[:split_idx]
        temp_test_labels = temp_labels[split_idx:]

        # join temp training and test sets for each Simpsons character together
        train_images = train_images + temp_train_images
        test_images = test_images + temp_test_images

        train_labels = train_labels + temp_train_labels
        test_labels = test_labels + temp_test_labels

    # normalize images
    train_images = np.array(train_images).astype(np.float32) / 255.0
    test_images = np.array(test_images).astype(np.float32) / 255.0

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # create validation set
    midpoint = int(len(test_images) / 2)

    val_images = test_images[:midpoint]
    test_images = test_images[midpoint:]

    val_labels = test_labels[:midpoint]
    test_labels = test_labels[midpoint:]

    # verify shape of data
    print(f'Training images shape: {train_images.shape}')
    print(f'Training labels shape: {train_labels.shape}')
    print(f'Validation images shape: {val_images.shape}')
    print(f'Validation labels shape: {val_labels.shape}')
    print(f'Test images shape: {test_images.shape}')
    print(f'Test labels shape: {test_labels.shape}')

    input_shape = (64, 64, 3)

    """
    # convert from array to image
    x = train_images[2000]
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
        x=train_images,
        y=train_labels,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=train_images.shape[0] // BATCH_SIZE,
        validation_data=(val_images, val_labels)
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
    print(f'\n# ----- TEST ACCURACY ----- #')
    test_loss, test_accuracy = model.evaluate(
        x=test_images,
        y=test_labels,
        verbose=0
    )
    print(f'Test accuracy: {test_accuracy:.6f}')

    # save the entire model (for later use in Android app)
    model.save(output_dir + "\\saved_model.h5")

    # ----- PREDICT ----- #
    last_idx = len(test_images)-1  # Homer 2245 (last pic)

    """
    # convert from array to image
    x = test_images[r]
    x = x * 255
    x = x.astype(np.uint8)
    Image.fromarray(x).show()
    quit()
    """

    print(f'\n# ----- GROUND TRUTH ----- #')
    print(f'Test label {last_idx} as int: {test_labels[last_idx]}')  # ground truth as int
    print(f'Test label {last_idx} as text: {num2char[test_labels[last_idx]]}')  # ground truth as text

    i = test_images[last_idx]  # single image
    i = np.expand_dims(i, 0)  # reshape: (1, 64, 64, 3)
    #print(i.shape)

    prediction = model.predict(i)  # predict on image

    print(f'\n# ----- PREDICTIONS ----- #')
    # prediction as distribution
    prediction_distribution = prediction
    print(f'Prediction as distribution: {prediction_distribution}')

    # prediction as int
    prediction = np.argmax(prediction_distribution)
    print(f'Prediction as numerical category: {prediction}')

    # prediction as text
    prediction = num2char[prediction]
    print(f'Prediction as text: {prediction}')

    # create prediction text
    print(f'\n# ----- PREDICTION TEXT BLOCK ----- #')
    t = []

    for i in range(num_classes):
        x = {"name": num2char[i],
             "value": prediction_distribution[0][i]
             }
        t.append(x)

    #print(t)

    # sort predictions in descending order
    t = sorted(t, key=lambda i: (i["value"]), reverse=True)

    #print(t)

    # build text block over image
    z = []
    for i in t:
        x = f'{i["name"]}: {i["value"]:.6f}'
        z.append(x)

    z = "\n".join(z)
    print(f'{z}')

    # write prediction text over image
    image_path = os.path.join(os.getcwd(), "data\\Training\\Homer Simpson\\pic_2245.jpg")
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 30)
    draw.text((0, 0), z, font=font)
    image.save(output_dir + "\\pred_image.png")


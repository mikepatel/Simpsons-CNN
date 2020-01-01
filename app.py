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
import numpy as np
import shutil
import glob
import imageio
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import tensorflow as tf


################################################################################
# delete a given directory
def delete_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)


################################################################################
# Main
if __name__ == "__main__":
    # 'temp' directory
    temp_dir = os.path.join(os.getcwd(), "temp")

    # delete 'predictions' sub-directory
    predictions_dir = os.path.join(temp_dir, "predictions")
    delete_dir(predictions_dir)

    # create 'predictions' sub-directory
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # mapping between Simpsons character and a numeric categorical label
    char2num = {
        "Bart Simpson": 0,
        "Homer Simpson": 1
    }

    # reverse mapping
    num2char = {v: k for k, v in char2num.items()}

    # number of output classes
    num_classes = len(char2num)

    # ----- PREDICT ----- #
    # load model
    filepath = os.path.join(temp_dir, "saved_model.h5")
    model = tf.keras.models.load_model(filepath)

    # read in test dataset
    homer_test_dir = os.path.join(os.getcwd(), "data\\Test\\Homer Simpson")
    image_files_pattern = homer_test_dir + "\\*.jpg"
    filenames = glob.glob(image_files_pattern)

    # preprocess images
    images = []

    for f in filenames:
        image = Image.open(f)

        # resize image
        resized_image = image.resize((64, 64))
        images.append(np.array(resized_image))

    # normalize images
    images = np.array(images).astype(np.float32) / 255.0

    # predict using model
    #images = np.expand_dims(images, 0)
    predictions = model.predict(images)
    #print(predictions)

    text = []
    for i in range(len(predictions)):
        image_text = []

        for j in range(num_classes):
            t = {
                "name": num2char[j],
                "value": predictions[i][j]
            }
            image_text.append(t)

        # sort predictions in descending order
        image_text = sorted(image_text, key=lambda x: (x["value"]), reverse=True)

        # build image text as single string
        z = []
        for d in image_text:
            x = f'{d["name"]}: {d["value"]:.6f}'
            z.append(x)

        z = "\n".join(z)

        text.append(z)

    #print(text)

    # write prediction text onto image
    for i in range(len(filenames)):
        # get image filename
        name = filenames[i].split("\\")[-1]

        image = Image.open(filenames[i])
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 8)
        draw.text((0, 0), text[i], font=font)
        image.save(predictions_dir + "\\pred_" + name)

    # create gif

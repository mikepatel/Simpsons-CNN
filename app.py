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
from random import shuffle
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
    """
    # 'temp' directory
    temp_dir = os.path.join(os.getcwd(), "temp")

    # delete 'predictions' sub-directory
    predictions_dir = os.path.join(temp_dir, "predictions")
    delete_dir(predictions_dir)
    """

    # create 'predictions' directory
    predictions_dir = os.path.join(os.getcwd(), "predictions")
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
    filepath = os.path.join(predictions_dir, "saved_model.h5")
    model = tf.keras.models.load_model(filepath)

    # read in images and predict one-by-one
    for character in char2num:
        image_files_pattern = os.path.join("data\\Test\\" + character) + "\\*.jpg"
        filenames = glob.glob(image_files_pattern)

        # for each image in character sub-directory
        for f in filenames:
            image = Image.open(f)
            original_image = image

            # resize image
            image = image.resize((64, 64))

            # normalize image
            image = np.array(image).astype(np.float32) / 255.0

            # reshape: (1, 64, 64, 3)
            image = np.expand_dims(image, 0)

            # model prediction
            prediction = model.predict(image)
            #print(f'Prediction: {prediction}')

            """
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
            """

            # create prediction text
            #print(f'\n# ----- PREDICTION TEXT BLOCK ----- #')
            text = []

            for i in range(num_classes):
                x = {"name": num2char[i],
                     "value": prediction[0][i]
                     }
                text.append(x)

            # sort predictions in descending order
            text = sorted(text, key=lambda i: (i["value"]), reverse=True)

            # build text block over image
            z = []
            for i in text:
                x = f'{i["name"]}: {i["value"]:.6f}'
                z.append(x)

            z = "\n".join(z)
            #print(f'{z}')

            # write prediction text over image
            # get image filename
            name = f.split("\\")[-1]

            #image_path = os.path.join(os.getcwd(), "data\\Training\\Homer Simpson\\pic_2245.jpg")
            #image = Image.open(image_path)
            draw = ImageDraw.Draw(original_image)
            font = ImageFont.truetype("arial.ttf", 8)
            draw.text((0, 0), z, font=font)
            original_image.save(predictions_dir + "\\pred_" + name)

    # ----- VISUALIZATION ----- #
    # create gif
    gif_filename = os.path.join(predictions_dir, "predictions.gif")

    # get all predicted images
    image_files_pattern = predictions_dir + "\\*.jpg"
    filenames = glob.glob(image_files_pattern)

    # shuffle filenames
    shuffle(filenames)

    # write all images to gif
    with imageio.get_writer(gif_filename, mode="I", fps=0.8) as writer:  # 'I' for multiple images
        for f in filenames:
            image = imageio.imread(f)
            writer.append_data(image)

    # delete all individual predicted images
    for f in filenames:
        if f.endswith(".jpg"):
            os.remove(f)

"""
Michael Patel
January 2019

Python 3.6.5
TensorFlow 2.0.0

Project description:
    Classifier for characters from The Simpsons

File description:
    Deploy application for Android mobile

"""
################################################################################
# Imports
import os
import numpy as np
import glob
from PIL import Image

import tensorflow as tf


################################################################################
# Main
if __name__ == "__main__":
    saved_model_dir = os.path.join(os.getcwd(), "saved_model")
    model_filepath = os.path.join(saved_model_dir, "converted_model.tflite")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=saved_model_dir)
    tflite_model = converter.convert()
    open(model_filepath, "wb").write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=model_filepath)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    print(input_details)
    output_details = interpreter.get_output_details()

    input_shape=input_details[0]["shape"]

    # get data
    image_files_pattern = os.path.join("data\\Test\\" + "Bart Simpson") + "\\*.jpg"
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

        input_data = image

        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)




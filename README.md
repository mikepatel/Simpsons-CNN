# Simpsons-CNN
## Overview
* CNN model to classify characters from The Simpsons
* Trained using TensorFlow 2.0
* Run inference on an Android device

## Data
* The dataset is composed of images of characters from The Simpsons provided by [alexattia](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)

## Methodology
* Using [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) to augment training data set
* Using [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) as the base model
* Using [TensorFlow Lite Converter](https://www.tensorflow.org/lite/convert) to convert trained model for use in an Android app

## Results
![gif](https://github.com/mikepatel/Simpsons-CNN/blob/master/saved/predictions/predictions.gif)

| Image | Prediction |
:------:|:-----------:
![](https://github.com/mikepatel/Simpsons-CNN/blob/master/data/test/bart_simpson_25.jpg) | ![](https://github.com/mikepatel/Simpsons-CNN/blob/master/saved/predictions/pred_bart_simpson_25.jpg)
![](https://github.com/mikepatel/Simpsons-CNN/blob/master/data/test/homer_simpson_13.jpg) | ![](https://github.com/mikepatel/Simpsons-CNN/blob/master/saved/predictions/pred_homer_simpson_13.jpg)
![](https://github.com/mikepatel/Simpsons-CNN/blob/master/data/test/homer_simpson_37.jpg) | ![](https://github.com/mikepatel/Simpsons-CNN/blob/master/saved/predictions/pred_homer_simpson_37.jpg)
![](https://github.com/mikepatel/Simpsons-CNN/blob/master/data/test/lisa_simpson_20.jpg) | ![](https://github.com/mikepatel/Simpsons-CNN/blob/master/saved/predictions/pred_lisa_simpson_20.jpg)

### Training after 15 epochs
![training plots](https://github.com/mikepatel/Simpsons-CNN/blob/master/saved/plots.png)

### Training after 15 epochs + 15 epochs of fine-tuning
![fine_tune](https://github.com/mikepatel/Simpsons-CNN/blob/master/saved/plots_finetune.png)

# Simpsons-CNN
## Overview
* CNN model to classify characters from The Simpsons
* Trained using TensorFlow 2.0
* Run inference on Android mobile platform, developed from TF integration with Android Studio Canary

## Data
* The [dataset](https://github.com/mikepatel/Simpsons-CNN/tree/master/data) is composed of images of characters from The Simpsons provided by [alexattia](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)

## File descriptions
* [app.py](https://github.com/mikepatel/Simpsons-CNN/blob/master/app.py) To run the trained model
* [deploy_mobile.py](https://github.com/mikepatel/Simpsons-CNN/blob/master/deploy_mobile.py) To create a TensorFlow Lite version of the trained model for use in an Android app
* [model.py](https://github.com/mikepatel/Simpsons-CNN/blob/master/model.py) For model definitions
* [parameters.py](https://github.com/mikepatel/Simpsons-CNN/blob/master/parameters.py) For constants and model parameters
* [train.py](https://github.com/mikepatel/Simpsons-CNN/blob/master/train.py) For preprocessing and training algorithm

## Instructions
### Train model
```
python train.py
```
### Use model for inference
```
python app.py
```

## Results
### Preliminary results after 5 epochs
| Image | Prediction |
:------:|:-----------:
![image](https://github.com/mikepatel/Simpsons-CNN/blob/master/data/Training/Homer%20Simpson/pic_2080.jpg) | ![Prediction](https://github.com/mikepatel/Simpsons-CNN/blob/master/results/18-12-2019_19-18-01/pred_image.png)

### Preliminary results after 8 epochs
![predictions gif](https://github.com/mikepatel/Simpsons-CNN/blob/master/predictions/predictions.gif)

## Training Visualization
![Training](https://github.com/mikepatel/Simpsons-CNN/blob/master/results/18-12-2019_19-18-01/Training%20Accuracy.png)

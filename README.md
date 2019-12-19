# Simpsons-CNN
## Overview
* CNN model to detect characters from The Simpsons
* Trained using TensorFlow 2.0

## Data
* The [dataset](https://github.com/mikepatel/Simpsons-CNN/tree/master/data) is composed of images of characters from The Simpsons provided from [alexattia](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)

## File descriptions
* [app.py](https://github.com/mikepatel/Simpsons-CNN/blob/master/app.py) To run the trained model
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
### Preliminary results after 50 epochs
| Image | Prediction |
:------:|:-----------:
![image](https://github.com/mikepatel/Simpsons-CNN/blob/master/data/Homer%20Simpson/pic_2080.jpg) | ![Prediction]()

## Training Visualization
![Training]()

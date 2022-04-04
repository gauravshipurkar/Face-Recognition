# Face-Recognition

## Table Of Content

- [Overview](#overview)
- [Motivation](#motivation)
- [Technical Aspect](#technical-aspect)
- [Installation](#installation)
- [To Do](#to-do)

## Overview

This project comprises face detection & recognition. Here, I have considered **Haar-Cascade Classifiers** for face detection & a **Convolutional Neural Network** for face recognition. The classification task is performed on a dataset of 5 people. 

## Motivation

We all know how time-consuming it is to take the attendance of students in the class manually. If the process is automated we can save the hassle, teachers have to go through every day taking the attendance of the students. Using this as a problem statement, we can make use of a database to store the names of students and we can use utilities like Face Detection & Recognition for further detection & corroboration of whether a student is present in the class.


## Technical Aspect 

For the classification task, I have collected a dataset of about 1000 images altogether. The images represent 5 people each having about 200 images. 

The libraries utilized are **TensorFlow, Keras, Media-Pipes & NumPy**. I have made use of the Media-Pipes library for the elegant look at the time of face detection. Further for the classification task, I had trained the **ResNet-50** model on the images setting only the last few layers as trainable. It leads to a problem of over-fitting. So, I scrapped this idea and trained a custom **CNN** model for the recognition/classification task. In this procedure, the bounding box acquired around the face of a person by **Haar-Cascades** is cropped utilizing the **Open-CV** library and passed into the **CNN** model for recognition of the person. 

![](#https://github.com/gauravshipurkar/Face-Recognition/blob/main/result.png)

## Installation

The Code is written in Python 3.8. If you don't have Python installed you can find it [here](#https://www.python.org/downloads/release/python-380/). To install **Open-CV** you can go [here]. To install **Media-Pipes** you can go [here]. To install the required packages and libraries, run this command in the project directory after cloning the repository:

```
pip install -r requirements.txt

```
## To Do

- Front-end for the project
- Simple database for the project
- Integration task

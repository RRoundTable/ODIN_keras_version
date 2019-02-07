from __future__ import print_function

import numpy as np
import time
from scipy import misc
from tensorflow.python.keras import layers
from tensorflow.python import keras



import calMetric as m
import calData2 as d
import tensorflow as tf

# CUDA_DEVICE = 0

start = time.time()
# loading data sets

# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Densenet trained on WideResNet-10:    wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
# nnName = "densenet10"

# imName = "Imagenet"


criterion = "categorical-crossentropy"

def train():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    input_shape = x_train.shape[1:]
    y_train = keras.utils.to_categorical(y_train, 10)
    input_tensor = layers.Input(shape=input_shape)
    net1 = keras.applications.densenet.DenseNet121(input_tensor=input_tensor, classes=10, weights=None)
    net1.compile(optimizer=tf.train.AdamOptimizer(0.001),
                 loss="categorical_crossentropy",
                 metrics=['accuracy'])
    net1.fit(x_train, y_train, batch_size=32, epochs=3)
    net1.save('densenet121_cifar10.h5')

    return net1

def test(nnName, dataName, CUDA_DEVICE, epsilon, temperature):

    net1 = keras.models.load_model('densenet121_cifar10.h5')  # 저장된 모델 load

    if dataName != "Uniform" and dataName != "Gaussian":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        testloaderOut=(x_test, y_test)

    if nnName == "CIFAR-10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        testloaderIn=(x_test, y_test)

    if nnName == "densenet10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        testloaderIn=(x_test, y_test)

    # dataname
    if dataName == "Gaussian":
        d.testGaussian(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, nnName, dataName, epsilon, temperature)
        m.metric(nnName, dataName)

    elif dataName == "Uniform":
        d.testUni(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, nnName, dataName, epsilon, temperature)
        m.metric(nnName, dataName)
    else:
        d.testData(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderOut, nnName, dataName, epsilon, temperature)
        m.metric(nnName, dataName)

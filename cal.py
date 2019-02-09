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
    print("--start training!--")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    input_shape = x_train.shape[1:]
    classes=10
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    input_tensor = layers.Input(shape=input_shape)

    net1 = keras.applications.densenet.DenseNet121(input_tensor=input_tensor, classes=10,weights=None) # return Model : transfer learning을 시도할 때 유연하게 대처하기 힘들다
    net1.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                 loss="categorical_crossentropy",
                 metrics=['accuracy'])
    train_len=len(x_train)
    x_train, y_train=x_train[:int(train_len*0.8)],y_train[:int(train_len*0.8)]
    x_val, y_val=x_train[int(train_len*0.8):],y_train[int(train_len*0.8):]
    net1.fit(x_train, y_train, batch_size=64, epochs=20,validation_data=(x_val,y_val))
    print("----------save----------")
    net1.save('densenet121_cifar10.h5')
    print("------test------")
    result=net1.evaluate(x_test,y_test,batch_size=100)
    print("loss : {}".format(result[0])) # loss
    print("accuracy : {}".format(result[1])) # accuracy

def test(nnName, dataName,epsilon, temperature):
    """
    :param nnName: in-distribution
    :param dataName: out-of-distribution
    :param epsilon: noiseMagnitude
    :param temperature: scaling
    """
    print("--start testing!--")
    net1 = keras.models.load_model('densenet121_cifar10.h5')  # 저장된 모델 load

    if dataName =="CIFAR-100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        testloaderOut=(x_test, y_test)

    if nnName == "CIFAR-10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        testloaderIn=(x_train[:100000], y_train[:100000])


    d.testData(net1,testloaderIn, testloaderOut, nnName, dataName, epsilon, temperature)
    m.metric(nnName, dataName)

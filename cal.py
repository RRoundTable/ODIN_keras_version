from __future__ import print_function

import numpy as np
import time
from scipy import misc
from tensorflow.python.keras import layers
from tensorflow.python import keras
from matplotlib.image import imread
from PIL import Image
from matplotlib.image import *
import os
from cifar10vgg import cifar10vgg



import calMetric as m
import calData as d
import tensorflow as tf


start = time.time()

def train():
    print("--start training!--")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    classes=10
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(save_best_only=True, filepath=os.path.join("./checkpoint","weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
                                           monitor="val_loss")
    ]

    try :
        print("저장된 모델 불러오는 중")
        net1 = cifar10vgg(train=False).model
        net1.compile(optimizer=tf.train.RMSPropOptimizer(0.0005),
                     loss=tf.losses.softmax_cross_entropy,
                     metrics=['accuracy'])
    except:
        print("저장된 모델이 없습니다")
        net1 = cifar10vgg(train=True).model
        net1.compile(optimizer=tf.train.RMSPropOptimizer(0.0005),
                     loss=tf.losses.softmax_cross_entropy,
                     metrics=['accuracy'])

    net1.fit(x_train, y_train, batch_size=32,epochs=300,validation_split=0.2, callbacks=callbacks)
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
    net1=cifar10vgg(train=False).model

    if nnName == "densenet10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # testloaderIn=x_train[:10000]
        testloaderIn = x_train[:10000]
    if dataName =="Imagenet_crop":
        # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        testloaderOut=load_images_from_folder("./Imagenet/test")

    if dataName =="CIFAR-100":
        (_, _), (testloaderOut, _) = tf.keras.datasets.cifar100.load_data()

    if dataName =="Gaussian":
        testloaderOut=np.random.standard_normal(size=testloaderIn.shape)+0.5
    if dataName =="Uniform":
        testloaderOut = np.random.uniform(0, 1, size=testloaderIn.shape)


    testloaderIn, testloaderOut= normalize(testloaderIn,testloaderOut)
    testloaderIn=(testloaderIn, y_train[:10000])

    d.testData(net1,testloaderIn, testloaderOut, nnName, dataName, epsilon, temperature)
    m.metric(nnName, dataName,temperature,epsilon)


def load_images_from_folder(folder):
    images=[]
    for filename in os.listdir(folder):
        img=Image.open(os.path.join(folder,filename))
        img=img.resize((32,32))
        img=img.copy()
        img=np.asarray(img)
        images.append(img)

    return np.asarray(images)

def normalize(X_train,X_test):
    #this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test
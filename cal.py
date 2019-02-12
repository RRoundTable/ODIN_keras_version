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




import calMetric as m
import calData2 as d
import tensorflow as tf



start = time.time()

def train():
    print("--start training!--")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    input_shape = x_train.shape[1:]

    classes=10
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    input_tensor = layers.Input(shape=input_shape)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint(save_best_only=True, filepath=os.path.join("./checkpoint","weights.{epoch:02d}-{val_loss:.2f}.hdf5"),
                                           monitor="val_loss")
    ]

    try :
        print("저장된 모델 불러오는 중")
        net1 = keras.models.load_model('densenet121_cifar10.h5')
        net1.compile(optimizer=tf.train.RMSPropOptimizer(0.0005),
                     loss=tf.losses.softmax_cross_entropy,
                     metrics=['accuracy'])
    except:
        print("저장된 모델이 없습니다")
        net1 = keras.applications.densenet.DenseNet121(input_tensor=input_tensor, classes=10,
                                                       weights=None)  # return Model : transfer learning을 시도할 때 유연하게 대처하기 힘들다
        net1.compile(optimizer=tf.train.AdamOptimizer(0.0003),
                     loss=tf.losses.softmax_cross_entropy,
                     metrics=['accuracy'])
    train_len=len(x_train)
    #x_train, y_train=x_train[:int(train_len*0.8)],y_train[:int(train_len*0.8)]
    #x_val, y_val=x_train[int(train_len*0.8):],y_train[int(train_len*0.8):]
    net1.fit(x_train, y_train, batch_size=32,epochs=100,validation_split=0.2, callbacks=callbacks)
    print("----------save----------")
    net1.save('densenet121_cifar10_190211.h5')
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
    net1 = keras.models.load_model('densenet121_cifar10_190211.h5')  # 저장된 모델 load

    if dataName =="Imagenet_crop":
        # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        testloaderOut=load_images_from_folder("./Imagenet/test")


    if nnName == "densenet10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        # testloaderIn=x_train[:10000]
        testloaderIn=x_train[:10000]

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
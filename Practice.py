from __future__ import print_function

import numpy as np
import time
from scipy import misc
from tensorflow.python.keras import layers
from tensorflow.python import keras
import tensorflow as tf
import numpy as np
import cv2
from matplotlib.image import imread
import os
from cifar10vgg import cifar10vgg
from PIL import Image
def train():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    input_shape = x_train.shape[1:]
    y_train = keras.utils.to_categorical(y_train, 10)
    input_tensor = layers.Input(shape=input_shape)
    net1 = keras.applications.densenet.DenseNet121(input_tensor=input_tensor, classes=10, weights=None)
    net1.compile(optimizer=tf.train.AdamOptimizer(0.001),
                 loss="categorical_crossentropy",
                 metrics=['accuracy'])
    net1.fit(x_train, y_train, batch_size=32, epochs=30)
    net1.save('densenet121_cifar10.h5')

    return net1



def model_check():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    net1 = cifar10vgg(train=False)
    net1=net1.model
    #net1=keras.models.load_model("densenet169_cifar10.h5")
    # compile을 한 후 evaluate를 해야한다
    net1.compile(optimizer=tf.train.AdamOptimizer(0.001),
                 loss="categorical_crossentropy",
                 metrics=['accuracy'])
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train, x_test=normalize(x_train,x_test)
    # for train data
    result=net1.evaluate(x_train,y_train, batch_size=100)
    print("train data :loss {} accuracy {}".format(result[0],result[1]))

    # for test data
    result = net1.evaluate(x_test, y_test, batch_size=100)
    print("test data :loss {} accuracy {}".format(result[0], result[1]))

    # 현재 overfiting 상태 : 어떻게 개선할지


def test():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    net1=keras.models.load_model("densenet121_cifar10.h5")
    # compile을 한 후 evaluate를 해야한다
    net1.compile(optimizer=tf.train.AdamOptimizer(0.001),
                 loss="categorical_crossentropy",
                 metrics=['accuracy'])
    y_test=keras.utils.to_categorical(y_test,10)

    input_shape = x_train.shape[1:]

    # loss, metric
    input_tensor=keras.backend.variable(np.expand_dims(x_test[0],0), dtype=tf.float32)
    # input_tensor = layers.Input(shape=input_shape,tensor=input_tensor)
    result=net1.evaluate(np.expand_dims(x_test[0],0),np.expand_dims(y_test[0],0), batch_size=1)

    # estimator=keras.estimator.model_to_estimator(net1)
    #
    #print(estimator.params)

    print(result) # loss , metric


    model=net1
    # gradient : keras.backend.function에 대해서 더 알아보기
    gradient=keras.backend.gradients(model.outputs, model.inputs)
    data=np.expand_dims(x_test[1],0)
    label=np.expand_dims(y_test[1],0)
    # gradient=sess.run(gradient, feed_dict={model.inputs :data , model.outputs: label})
    gradient=keras.backend.function(model.inputs,gradient)
    gradient=gradient([data])
    print("gradient" ,np.array(gradient)[0,0,:,:])

    gradient = np.where(np.array(gradient) > 0, np.array(gradient), -1)
    print("gradient2", np.array(gradient)[0, 0, :, :])
    gradient = np.where(np.array(gradient) < 0.0, np.array(gradient), 1)

    print("gradient3", np.array(gradient)[0, 0, :, :])
    print("mask {}".format(gradient[0,0,:,:]))
    #print(np.isin(gradient,0)) # 모두 0 잘못된 gradient
    print("max gradient : ",np.max(gradient))
    print(np.array(gradient).astype("float32")-.5)
    print((np.array(gradient).astype("float32")-.5)*2)

    #print(np.isin(gradient,0).astype("int32")) # normalize {0,1}

def data():

    x_test=load_images_from_folder("./Imagenet/test")
    print("x_test shape :",np.asarray(x_test).shape)



    # gradient값 표준화 지표
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    testloader=(x_train, y_train)
    data=np.mean(x_train,axis=0)
    data = np.mean(data, axis=0)
    data = np.mean(data, axis=0)
    print(np.mean(x_train, axis=0).shape) # [125.30691805 122.95039414 113.86538318]
    print(data)

    for img, label in zip(x_train, y_train):
        if np.squeeze(label)==1:
            print(label)
            cv2.resize(img, (300,300))
            cv2.imshow(str(label),img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_images_from_folder(folder):
    images=[]
    for filename in os.listdir(folder):
        img=Image.open(os.path.join(folder,filename))
        img=img.resize((32,32))
        img=img.copy()
        img=np.asarray(img)
        images.append(img)
    return images

def predict():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    net1=keras.models.load_model("densenet121_cifar10.h5")
    # compile을 한 후 evaluate를 해야한다
    net1.compile(optimizer=tf.train.AdamOptimizer(0.001),
                 loss="categorical_crossentropy",
                 metrics=['accuracy'])
    y_test=keras.utils.to_categorical(y_test,10)

    input_shape = x_train.shape[1:]

    # loss, metr"ic

    # input_tensor = layers.Input(shape=input_shape,tensor=input_tensor)

    # estimator=keras.estimator.model_to_estimator(net1)
    #
    #print(estimator.params)


    # predict
    predict=net1.predict(x_test[:2],batch_size=2)
    print(predict) # softmax value
    print(np.array(predict).shape)

    print("predict max", np.array(predict).max(axis=1))



    model=keras.models.clone_model(net1)
    # gradient : keras.backend.function에 대해서 더 알아보기
    gradient=keras.backend.gradients(model.outputs, model.inputs)

    print("-----------------------------------------------------")
    print(gradient)

    # data=np.expand_dims(x_test[:2],0)
    # label=np.expand_dims(y_test[:2],0)
    data, label= x_test[:2], y_test[:2]
    # gradient=sess.run(gradient, feed_dict={model.inputs :data , model.outputs: label})
    gradient=keras.backend.function(model.inputs,gradient)
    gradient=gradient([data])
    print("gradient shape : ", np.array(gradient).shape)
    print("gradient shape2 : ",np.squeeze(gradient).shape)
    print("input shape : ",data.shape)
    mask=np.isin(gradient,0)
    #print(np.isin(gradient,0)) # 모두 0 잘못된 gradient
    print("max gradient : ",np.max(gradient))

    gradient=np.squeeze(gradient)


    print("gradient final {}".format(gradient[0].shape))
    print("gradient final {}".format(gradient[:][:][:][:].shape))
    print("gradient final _________ {}".format(gradient[0][0:32][0:32][0].shape)) # 32 x 3 ? 차례대로 적용되서
    # print("gradient final {}".format(gradient[:][:][:][1].shape))
    # print("gradient final {}".format(gradient[:][:][:][1].shape))
    gradient[:,:,:0]= ( gradient[:,:,:0]) / (63.0 / 255.0)
    gradient[0][:][:][1] = (gradient[0][:][:][1]) / (62.1 / 255.0)
    gradient[0][:][:][2] = (gradient[0][:][:][2]) / (66.7 / 255.0)
    print(gradient)
    tempInput = np.add(data, gradient)
    print("tempinput {}".format(tempInput.shape))
    outputs = net1.predict(tempInput)

    nnOutputs = outputs - np.max(outputs)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

    #print(np.isin(gradient,0).astype("int32")) # normalize {0,1}


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

if __name__=="__main__":
    model_check()
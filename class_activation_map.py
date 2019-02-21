from __future__ import print_function
from tensorflow.python import keras
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.applications import VGG16
import tensorflow as tf
import os
import cv2
from PIL import Image


from cifar10vgg import cifar10vgg

config=tf.ConfigProto()
config.gpu_options.allow_growth=True

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

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)
x_train, x_test=normalize(x_train,x_test)

def Resize(x):
    """
    Resize images to size using bicubic interpolation.
    :param x: input
    :return:
    """
    y=tf.image.resize_bicubic(x,size=(256,256))
    return y


def create_model():
    print("create model ...")
    inputs=Input(shape=(32,32,3))
    resize=Lambda(Resize,(256,256,3))(inputs)
    base_model=VGG16(include_top=False,pooling="avg")
    GAV=base_model(resize)
    #GAV=GlobalAveragePooling2D()(conv)
    outputs=Dense(10,activation='softmax')(GAV)
    model=Model(inputs,outputs)
    model.compile(optimizer='sgd',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])
    return model

def training(model):
    earlystopping = EarlyStopping()
    modelchenkpoint = ModelCheckpoint('activation_map.h5',save_best_only=True)
    model.fit(x_train,y_train,
              batch_size=16,
              nb_epoch=200,
              validation_split=0.2,
              callbacks=[earlystopping,modelchenkpoint])
    print("model training END !")


def get_conv(model, test_imgs):
    print("geting conv ...")
    inputs=Input(shape=(32,32,3))
    vgg=model.layers[2].layers[:-1] # layer name
    vgg=Sequential(vgg) # model
    print(model.layers[2].layers[-2].name)
    resize=Lambda(Resize,(256,256,3))(inputs)
    outputs=vgg(resize)
    print("output : {}".format(outputs))
    new_model=Model(inputs,outputs)
    print('Loading the conv_features of test_images .......')
    conv_features = new_model.predict(test_imgs)
    print('Loading the conv_features done!!!')
    print(conv_features.shape)
    return conv_features

def visualization_CAM(idx,dis,feature_map, weights,predict,img):

    # 0번째 example
    feature_map=np.squeeze(feature_map) # batch, height, width, channels
    img=cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite("./result/original_resize.png", img)
    cam=np.zeros(shape=feature_map[:,:,0].shape,dtype=np.float)
    for i in range(feature_map.shape[-1]):
        cam+=feature_map[:,:,i]*weights[i,predict]
    cam=cam/np.max(cam)
    cam=cv2.resize(cam, (256,256))

    heatmap= cv2.applyColorMap(np.uint8(255*cam),cv2.COLORMAP_JET)
    heatmap[np.where(cam<0.2)]=0
    img= heatmap*0.7+img
    cv2.imwrite("./result_{}/activation_map_{}_{}.png".format(dis,idx,dis),img)



def load_images_from_folder(folder):
    images=[]
    for filename in os.listdir(folder):
        img=Image.open(os.path.join(folder,filename))
        img=img.resize((32,32))
        img=img.copy()
        img=np.asarray(img)
        images.append(img)

    return np.asarray(images)


if __name__ == '__main__':

    model=create_model()
    if not os.path.exists('activation_map.h5'):
        print("start training ...")
        training(model)
    else:
        print("start loading weights ...")
        model.load_weights('activation_map.h5')

    #testloaderOut = load_images_from_folder("./Imagenet/test") # out-of-distribution
    (_, _), (testloaderOut, _) = tf.keras.datasets.cifar100.load_data()
    for i in range(100):
        # in-distribution
        idx=i
        conv_features=get_conv(model, np.expand_dims(x_test[idx],0))
        predict_label=model.predict(np.expand_dims(x_test[idx],0))
        predict_label=np.argmax(predict_label)

        print("Extraction the weight between GAV and dense")
        w=model.get_weights()[-2] # GAP outputs
        # visualization_CAM(idx,"in",conv_features,w,predict_label,x_test[idx])

        # out-of-distribution
        conv_features = get_conv(model, np.expand_dims(testloaderOut[idx], 0))
        predict_label = model.predict(np.expand_dims(testloaderOut[idx], 0))
        predict_label = np.argmax(predict_label)

        visualization_CAM(idx,"out", conv_features, w, predict_label,testloaderOut[idx])









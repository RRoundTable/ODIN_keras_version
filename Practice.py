from __future__ import print_function

import numpy as np
import time
from scipy import misc
from tensorflow.python.keras import layers
from tensorflow.python import keras
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np

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
    # predict
    predict=net1.predict(x_test[:2],batch_size=2)
    print(predict) # softmax value

    loss= result[0]
    print("loss {}".format(loss)) # loss 확인
    print(keras.models.clone_model(net1))
    model=keras.models.clone_model(net1)




    # gradient : keras.backend.function에 대해서 더 알아보기
    gradient=keras.backend.gradients(model.outputs, model.inputs)

    print("-----------------------------------------------------")
    print(gradient)

    print(model.inputs)

    print(model.outputs)
    sess=tf.Session()
    data=np.expand_dims(x_test[0],0)
    label=np.expand_dims(y_test[0],0)
    # gradient=sess.run(gradient, feed_dict={model.inputs :data , model.outputs: label})
    gradient=keras.backend.function(model.inputs,gradient)
    gradient=gradient([data])

    print(len(gradient)) # 1
    print(len(gradient[0])) # 1
    print(len(gradient[0][0])) # 32
    print(len(gradient[0][0][0])) # 32
    print(len(gradient[0][0][0][0])) # 3
    print("gradient shape : ", np.array(gradient).shape)
    print("gradient shape2 : ",np.squeeze(gradient).shape)
    print("input shape : ",data.shape)
    mask=np.isin(gradient,0)
    #print(np.isin(gradient,0)) # 모두 0 잘못된 gradient
    print("max gradient : ",np.max(gradient))

    #print(np.isin(gradient,0).astype("int32")) # normalize {0,1}

def data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    testloader=(x_train, y_train)

    for i, data in enumerate(zip(testloader[0],testloader[1])):
        print(i)
        print(len(data))
        print(data[0])
        print(data[1])

if __name__=="__main__":
    test()
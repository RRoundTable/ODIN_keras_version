
from __future__ import print_function

import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import utils
from tensorflow.python import keras
import tensorflow as tf
import time

def testData(net1,  testloaderIn, testloaderOut, nnName, dataName, noiseMagnitude1, temper):
    """
    :param net1: test model
    :param testloader10: in-distribution data
    :param testloader: out-of-distribution data
    :param nnName: in-distribution name
    :param dataName: out-of-distribution name
    :param noiseMagnitude1: noise
    :param temper: scaling
    """
    t0=-time.time()
    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
    N = len(testloaderIn[0])
    print(N)
    if dataName == "iSUN": N = 8925
    print("Processing in-distribution images")
    print("in-distribution : {}".format(nnName))
    ########################################################################################################
    # in-distribution
    images= testloaderIn[0] # in distribution
    labels=testloaderIn[1]
    labels=keras.utils.to_categorical(labels,10)
    batch_size=int(N/10)
    print("images : ",images.shape)
    for i in range(10):
        outputs=net1.predict(images[batch_size*i:(i+1)*batch_size],verbose=1 ,batch_size=batch_size)
        y_labels=labels[batch_size*i:(i+1)*batch_size]
        nnOutputs=outputs-np.expand_dims(np.max(outputs, axis=1),1) # (50,10) (50,1)


        nnOutputs=np.exp(nnOutputs)/np.expand_dims(np.sum(np.exp(nnOutputs),axis=1),1)
        print(nnOutputs.shape)


        #print(np.expand_dims(np.max(outputs, axis=1),1))
        for j in range(batch_size):
            if j==batch_size-1: print("--{} batch baseline in distribution--".format(i))
            f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs[j])))

        # # Normalizing the gradient to binary in {0, 1}
        y_true = keras.layers.Input(shape=[None,10])
        loss=keras.backend.mean(keras.backend.categorical_crossentropy(y_true,net1.output))#
        gradient=keras.backend.gradients(loss, net1.inputs) # loss, variable
        gradient=keras.backend.function(net1.inputs+[y_true], gradient) # input / output
        gradient=gradient([images[batch_size*i:(i+1)*batch_size],[y_labels]])

        # # Normalizing the gradient to binary in {0, 1}
        gradient = np.where(np.array(gradient) > 0, np.array(gradient), -1)
        gradient = np.where(np.array(gradient) < 0, np.array(gradient), 1)
        # #gradient=(gradient.astype("float32")-.5)*2
        #
        # # Normalizing the gradient to the same space of image
        gradient=np.squeeze(gradient)
        gradient[:,:,:,0] = (gradient[:,:,:,0]) /(63.0 / 255.0)
        gradient[:, :, :, 1] = (gradient[:,:,:,1]) /(62.1 / 255.0)
        gradient[:, :, :, 2] = (gradient[:,:,:,2]) /(66.7 / 255.0)

        tempInput=np.add(images[batch_size*i:(i+1)*batch_size],gradient*noiseMagnitude1*-1)
        outputs=net1.predict(tempInput,batch_size=batch_size)
        outputs = outputs / temper

        # Calculating the confidence atfer adding perturbations
        nnOutputs = outputs - np.expand_dims(np.max(outputs, axis=1), 1)  # (50,10) (50,)
        nnOutputs = np.exp(nnOutputs) / np.expand_dims(np.sum(np.exp(nnOutputs), axis=1), 1)
        print(nnOutputs.shape)
        for j in range(batch_size):
            if j == batch_size - 1:  print("--{} batch our in distribution--".format(i))
            g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs[j])))
        if i % 5 == 0:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(i+1-1000, N-batch_size*5, time.time()-t0))
            t0 = time.time()

    f1.close()
    g1.close()

    print("Processing out of distribution")
    print("out-of-distribution : {}".format(dataName))
    images=testloaderOut # out of distribution
    labels=np.random.randint(0,9,N)
    labels=keras.utils.to_categorical(labels,10)
    for i in range(10):
        outputs = net1.predict(images[batch_size * i:(i + 1) * batch_size], batch_size=batch_size)
        print("max value : ", np.expand_dims(np.max(outputs, axis=1), 1))  # max value
        nnOutputs = outputs - np.expand_dims(np.max(outputs, axis=1), 1)  # (50,10) (50,)
        nnOutputs = np.exp(nnOutputs) / np.expand_dims(np.sum(np.exp(nnOutputs), axis=1), 1)
        print(nnOutputs.shape)


        for j in range(batch_size):
            if j == batch_size - 1:print("--{} batch baseline out-of- distribution--".format(i))
            f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs[j])))

        # Get gradient w.r.t input
        y_true = keras.layers.Input(shape=[None, 10])
        loss = keras.backend.mean(keras.backend.categorical_crossentropy(y_true, net1.output))  #
        gradient = keras.backend.gradients(loss, net1.inputs)  # loss, variable
        gradient = keras.backend.function(net1.inputs + [y_true], gradient)  # input / output

        gradient = gradient([images[batch_size * i:(i + 1) * batch_size], [labels[batch_size * i:(i + 1) * batch_size]]])

        # Normalizing the gradient to binary in {0, 1}
        gradient = np.where(np.array(gradient) > 0, np.array(gradient), -1)
        gradient = np.where(np.array(gradient) < 0, np.array(gradient), 1)
        # gradient = (gradient.astype("float32") - .5) * 2
        #
        # # Normalizing the gradient to the same space of image : error
        gradient = np.squeeze(gradient)
        gradient[:, :, :, 0] = (gradient[:, :, :, 0]) /(63.0 / 255.0)
        gradient[:, :, :, 1] = (gradient[:, :, :, 1]) / (62.1 / 255.0)
        gradient[:, :, :, 2] = (gradient[:, :, :, 2]) / (66.7 / 255.0)
        tempInput = np.add(images[batch_size * i:(i + 1) * batch_size], gradient*noiseMagnitude1*-1)

        outputs = net1.predict(tempInput, batch_size=batch_size)
        # using Temperature scaling
        outputs = outputs / temper
        # Calculating the confidence atfer adding perturbations
        nnOutputs = outputs - np.expand_dims(np.max(outputs, axis=1), 1)  # (50,10) (50,)
        nnOutputs = np.exp(nnOutputs) / np.expand_dims(np.sum(np.exp(nnOutputs), axis=1), 1)
        print(nnOutputs.shape)
        for j in range(batch_size):
            if j == batch_size - 1: print("--{} batch Our out-of- distribution--".format(i))
            g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs[j])))
        if i % 5 == 0:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(i+1-1000, N-batch_size*5, time.time()-t0))
            t0 = time.time()


    f2.close()
    g2.close()










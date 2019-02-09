
from __future__ import print_function

import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import utils
from tensorflow.python import keras
import tensorflow as tf
import time

def testData(net1,  testloader10, testloader, nnName, dataName, noiseMagnitude1, temper):
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
    N = len(testloader10[0])
    if dataName == "iSUN": N = 8925
    print("Processing in-distribution images")
    print("in-distribution : {}".format(nnName))
    ########################################################################################################
    # in-distribution
    images, labels = testloader10 # in distribution
    labels=keras.utils.to_categorical(labels,10)
    batch_size=int(N/100)
    for i in range(100):
        outputs=net1.predict(images[batch_size*i:(i+1)*batch_size], batch_size=batch_size)
        nnOutputs=outputs-np.expand_dims(np.max(outputs, axis=1),1) # (50,10) (50,)
        nnOutputs=np.exp(nnOutputs)/np.expand_dims(np.sum(np.exp(nnOutputs),axis=1),1)
        print(nnOutputs.shape)
        for j in range(batch_size):
            if j==batch_size-1: print("--{} batch baseline in distribution--".format(i))
            f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs, axis=1)[j]))

        # Normalizing the gradient to binary in {0, 1}
        gradient=keras.backend.gradients(net1.outputs, net1.inputs) # loss, variable
        gradient=keras.backend.function(net1.inputs, gradient)
        gradient=gradient([images[batch_size*i:(i+1)*batch_size]])

        # Normalizing the gradient to binary in {0, 1}
        gradient=np.isin(gradient,0).astype("int32")
        gradient=(gradient.astype("float32")-.5)*2

        # Normalizing the gradient to the same space of image
        gradient=np.squeeze(gradient)
        gradient[:,:,:,0] = (gradient[:,:,:,0]) / (63.0 / 255.0)
        gradient[:, :, :, 1] = (gradient[:,:,:,1]) / (62.1 / 255.0)
        gradient[:, :, :, 2] = (gradient[:,:,:,2]) / (66.7 / 255.0)

        tempInput=np.add(images[batch_size*i:(i+1)*batch_size],gradient)
        outputs=net1.predict(tempInput,batch_size=batch_size)
        outputs = outputs / temper

        # Calculating the confidence atfer adding perturbations
        nnOutputs = outputs - np.expand_dims(np.max(outputs, axis=1), 1)  # (50,10) (50,)
        nnOutputs = np.exp(nnOutputs) / np.expand_dims(np.sum(np.exp(nnOutputs), axis=1), 1)
        print(nnOutputs.shape)
        for j in range(batch_size):
            if j == batch_size - 1:  print("--{} batch our in distribution--".format(i))
            g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs, axis=1)[j]))
        if i % 5 == 0:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(i+1-1000, N-batch_size*5, time.time()-t0))
            t0 = time.time()



    print("Processing out of distribution")
    print("out-of-distribution : {}".format(dataName))
    images, labels = testloader # out of distribution
    for i in range(100):
        outputs = net1.predict(images[batch_size * i:(i + 1) * batch_size], batch_size=batch_size)
        nnOutputs = outputs - np.expand_dims(np.max(outputs, axis=1), 1)  # (50,10) (50,)
        nnOutputs = np.exp(nnOutputs) / np.expand_dims(np.sum(np.exp(nnOutputs), axis=1), 1)
        print(nnOutputs.shape)
        for j in range(batch_size):
            if j == batch_size - 1:print("--{} batch baseline out-of- distribution--".format(i))
            f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs, axis=1)[j]))

        # Normalizing the gradient to binary in {0, 1}
        gradient = keras.backend.gradients(net1.outputs, net1.inputs)  # loss, variable
        gradient = keras.backend.function(net1.inputs, gradient)
        gradient = gradient([images[batch_size * i:(i + 1) * batch_size]])

        # Normalizing the gradient to binary in {0, 1}
        gradient = np.isin(gradient, 0).astype("int32")
        gradient = (gradient.astype("float32") - .5) * 2

        # Normalizing the gradient to the same space of image : error
        gradient = np.squeeze(gradient)
        gradient[:, :, :, 0] = (gradient[:, :, :, 0]) / (63.0 / 255.0)
        gradient[:, :, :, 1] = (gradient[:, :, :, 1]) / (62.1 / 255.0)
        gradient[:, :, :, 2] = (gradient[:, :, :, 2]) / (66.7 / 255.0)

        tempInput = np.add(images[batch_size * i:(i + 1) * batch_size], gradient)
        outputs = net1.predict(tempInput, batch_size=batch_size)
        # using Temperature scaling
        outputs = outputs / temper
        # Calculating the confidence atfer adding perturbations
        nnOutputs = outputs - np.expand_dims(np.max(outputs, axis=1), 1)  # (50,10) (50,)
        nnOutputs = np.exp(nnOutputs) / np.expand_dims(np.sum(np.exp(nnOutputs), axis=1), 1)
        print(nnOutputs.shape)
        for j in range(batch_size):
            if j == batch_size - 1: print("--{} batch Our out-of- distribution--".format(i))
            g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs, axis=1)[j]))
        if i % 5 == 0:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(i+1-1000, N-batch_size*5, time.time()-t0))
            t0 = time.time()










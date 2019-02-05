
from __future__ import print_function

import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import utils
from tensorflow.python import keras
import tensorflow as tf
import time

def testData(net1, criterion, CUDA_DEVICE, testloader10, testloader, nnName, dataName, noiseMagnitude1, temper):
    t0=-time.time()

    if net1=="densenet121":
        input_shape=testloader10[0][0].shape[1:]
        input_tensor=layers.Input(shape=input_shape)
        net1=keras.applications.densenet.DenseNet121(input_tensor=input_tensor)

    f1 = open("./softmax_scores/confidence_Base_In.txt", 'w')
    f2 = open("./softmax_scores/confidence_Base_Out.txt", 'w')
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
    N = 100
    if dataName == "iSUN": N = 8925
    print("Processing in-distribution images")
    ########################################################################################################
    # in-distribution

    images, _ = testloader10
    for j, data in enumerate(zip(testloader10[0],testloader10[1])): # image하고 label하고 따로
        print("{} 번째".format(j))
        images, labels=data
        images=np.expand_dims(images,0)
        labels=keras.utils.to_categorical(labels,10)
        outputs=net1.predict(images, batch_size=1)
        nnOutputs=outputs-np.max(outputs)
        nnOutputs=np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # using Temperature scaling
        outputs=outputs/temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp=np.argmax(nnOutputs)
        labels=maxIndexTemp
        labels=keras.utils.to_categorical(labels,num_classes=10) # cifar1000
        #loss=net1.evaluate(images, labels, batch_size=1)

        # Normalizing the gradient to binary in {0, 1}
        gradient=keras.backend.gradients(net1.outputs, net1.inputs) # loss, variable
        gradient=keras.backend.function(net1.inputs, gradient)
        gradient=gradient([images])

        # Normalizing the gradient to binary in {0, 1}
        gradient=np.isin(gradient,0).astype("int32")
        gradient=(gradient.astype("float32")-.5)*2

        # Normalizing the gradient to the same space of image : error
        gradient=np.squeeze(gradient)
        gradient[0][:][:][0] = (gradient[0][:][:][0]) / (63.0 / 255.0)
        gradient[0][:][:][1] = (gradient[0][:][:][1]) / (62.1 / 255.0)
        gradient[0][:][:][2] = (gradient[0][:][:][2]) / (66.7 / 255.0)

        tempInput=np.add(images,gradient)
        outputs=net1.predict(tempInput)

        # Calculating the confidence atfer adding perturbations
        nnOutputs=outputs-np.max(outputs)
        nnOutputs=np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

        if j== N-1: break

    print("Processing out of distribution")
    for j, data in enumerate(zip(testloader[0],testloader[1])):
        print("{} 번재".format(j))
        images, labels=data
        images=np.expand_dims(images,0)
        labels=np.random.randint(low=0,high=9.,size=1)
        labels=keras.utils.to_categorical(labels,10)
        outputs=net1.predict(images, batch_size=1)
        print(outputs)
        nnOutputs=outputs-np.max(outputs)
        nnOutputs=np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

        # using Temperature scaling
        outputs=outputs/temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp=np.argmax(nnOutputs)
        labels=maxIndexTemp
        labels=keras.utils.to_categorical(labels,num_classes=10) # cifar1000
        #loss=net1.evaluate(images, labels, batch_size=1)

        # Normalizing the gradient to binary in {0, 1}
        gradient=keras.backend.gradients(net1.outputs, net1.inputs) # loss, variable
        gradient=keras.backend.function(net1.inputs, gradient)
        gradient=gradient([images])

        # Normalizing the gradient to binary in {0, 1}
        gradient=np.isin(gradient,0).astype("int32")
        gradient=(gradient.astype("float32")-.5)*2
        print(gradient.shape)
        gradient=np.squeeze(gradient)
        # Normalizing the gradient to the same space of image : 한번 손보기
        gradient[0][:][:][0] = (gradient[0][:][:][0]) / (63.0 / 255.0)
        gradient[0][:][:][1] = (gradient[0][:][:][1]) / (62.1 / 255.0)
        gradient[0][:][:][2] = (gradient[0][:][:][2]) / (66.7 / 255.0)

        tempInput = np.add(images, gradient)
        outputs = net1.predict(tempInput)

        # Calculating the confidence atfer adding perturbations
        nnOutputs=outputs-np.max(outputs)
        nnOutputs=np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
            t0 = time.time()

        if j== N-1: break








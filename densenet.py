"""DenseNet models for Keras.

# Reference paper
- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation
- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from . import get_submodules_from_kwargs
from . import imagenet_utils
from .imagenet_utils import decode_predictions
from .imagenet_utils import _obtain_input_shape

from tensorflow.python.keras import layers
from tensorflow.python.keras import backend
from tensorflow.python.keras import models
from tensorflow.python.keras import utils


BASE_WEIGTHS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/densenet/')
DENSENET121_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET121_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET169_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET201_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')



def dense_block(x,blocks,name):
    """
    A dense block.
        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.
        # Returns
            output tensor for the block.
     """

    for i in range(blocks):
        x=conv_block(x,32, name=name+"_block"+str(i+1))


    return x

def transition_block(x, reduction, name):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + "_bn")(x)
    x=layers.Activation("relu", name=name+"_relu")(x)
    x=layers.Conv2D(int(backend.int_shape(x)[bn_axis]*reduction),
                    1, use_bias=False, name=name+"_conv")(x)
    x=layers.AveragePooling2D(2, strides=2, name=name+"_pool")(x)
    return x

def conv_block(x,growth_rate,name):

    bn_axis=3 if backend.image_data_format()=='channels_last' else 1
    x1= layers.BatchNormalization(axis=bn_axis,
                                  epsilon=1.001e-5,
                                  name=name+"_0_bn")(x)
    x1=layers.Activation("relu", name=name+"_0_relu")(x1)
    x1=layers.Conv2D(4*growth_rate,1,use_bias=False,name=name+"_1_conv")(x1)
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + "_1_bn")(x1)
    x1 = layers.Activation("relu", name=name + "_1_relu")(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding="same",use_bias=False, name=name + "_2_conv")(x1)
    x=layers.Concatenate(axis=bn_axis, name=name+"_concat")([x,x1])
    return x


def DenseNet(blocks,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):

    # Determine proper input shape

    if input_tensor is None:
        img_input=layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input=layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input=input_tensor

    bn_axis= 3 if backend.image_data_format()=="channels_last" else 1

    x=layers.ZeroPadding2D(padding=((3,3),(3,3)))(img_input)
    x=layers.Conv2D(64,7,stride=2,use_bias=False, name="conv1/conv")(X)
    x=layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name='conv1/bn')(x)
    x=layers.Activation('relu', name= "conv1/relu")(x)
    x=layers.ZeroPadding2D(padding=((1,1),(1,1)))(x)
    x=layers.MaxPooling2D(3, strides=2, name="pool1")(x)

    x=dense_block(x,blocks[0],name="conv2")
    x=transition_block(x,0.5, name="pool2")
    x = dense_block(x, blocks[1], name="conv3")
    x = transition_block(x, 0.5, name="pool3")
    x = dense_block(x, blocks[2], name="conv4")
    x = transition_block(x, 0.5, name="pool4")
    x = dense_block(x, blocks[3], name="conv5")
    x = transition_block(x, 0.5, name="pool5")

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x=layers.Activation('relu', name="relu")(x)

    if include_top:
        x=layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x=layers.Dense(classes, activation='softmax', name="fc1000")(x)
    else:
        if pooling=="avg":
            x=layers.AveragePooling2D(name="avg_pool")(x)
        elif pooling=="max":
            x=layers.MaxPooling2D(name="max_pool")(x)

    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input



    # Create Model
    if blocks==[6,12,24,16]:
        model=models.Model(inputs, x,name="densenet121")
    elif blocks==[6,12,32,32]:
        model=models.Model(inputs, x,name="densenet169")
    elif blocks==[6,12,48,32]:
        model=models.Model(inputs, x,name="densenet201")
    else :
        model=models.Model(inputs, x,name="densenet")

    if weights == 'imagenet':
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = utils.get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET121_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='9d60b8095a5708f2dcce2bca79d332c7')
            elif blocks == [6, 12, 32, 32]:
                weights_path = utils.get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET169_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='d699b8f76981ab1b30698df4c175e90b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = utils.get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET201_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
        else:
            if blocks == [6, 12, 24, 16]:
                weights_path = utils.get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET121_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='30ee3e1110167f948a6b9946edeeb738')
            elif blocks == [6, 12, 32, 32]:
                weights_path = utils.get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET169_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = utils.get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET201_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model



def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 24, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)


def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 32, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)


def DenseNet201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 48, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)



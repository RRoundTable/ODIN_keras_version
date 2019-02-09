

from ODIN_keras import calMetric as m
import tensorflow as tf


def result(nnName, dataName):

    m.metric(nnName,dataName)


if __name__=="__main__":
    nnName="densenet10"
    dataName="CIFAR-100"
    result(nnName,dataName)


import calMetric as m
import tensorflow as tf

# 결과만 확인하고 싶다면
def result(nnName, dataName):

    m.metric(nnName,dataName,10000,0.0014)


if __name__=="__main__":
    nnName="densenet10"
    dataName="Imagenet_crop"
    result(nnName,dataName)
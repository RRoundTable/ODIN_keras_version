# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import misc


def tpr95(name):
    # calculate the falsepositive error when tpr is 95%
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 1
    if name == "CIFAR-100":
        start = 0.01
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):

        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.955 and tpr >= 0.945:
            fpr += error2
            total += 1
        if tpr<0.93: break
    fprBase = fpr / total

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 1000000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    total = 0.0
    fpr = 0.0
    print("x1 : {}".format(X1))
    print(end)
    for delta in np.arange(start, end, gap):

        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        print(tpr)
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))

        if tpr <= 0.955 and tpr >= 0.945:
            print(tpr)
            fpr += error2
            total += 1
        if tpr < 0.93:
            print("tpr95 끝")
            break
    fprNew = fpr / total

    return fprBase, fprNew


def auroc(name,temperature, noise):
    # calculate the AUROC
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 1
    if name == "CIFAR-100":
        start = 0.01
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    aurocBase = 0.0
    fprTemp = 1.0
    x_plot_base=[] # fpr
    y_plot_base=[] # tpr
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        x_plot_base.append(tpr)
        y_plot_base.append(fpr)
        aurocBase += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocBase += fpr * tpr
    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    aurocNew = 0.0
    fprTemp = 1.0
    x_plot_our = []  # fpr
    y_plot_our = []  # tpr
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        x_plot_our.append(tpr)
        y_plot_our.append(fpr)
        aurocNew += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocNew += fpr * tpr

    # save graph
    plt.figure()
    plt.plot(y_plot_base,x_plot_base, color="red", label="ROC Curve baseline", lw=2)
    plt.plot(y_plot_our,x_plot_our, color="blue", label="roc Curve Our", lw=2)
    plt.plot([0,1],[0,1], color='navy', linestyle='--')
    plt.xlabel("False Positive rate")
    plt.ylabel("True Positive rate")
    plt.xlim([0.0,1.0])
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.savefig("./graph/ROC_T_{}_M_{}.png".format(temperature,noise))



    return aurocBase, aurocNew


def auprIn(name,temperature, noise):
    # calculate the AUPR
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 1
    if name == "CIFAR-100":
        start = 0.01
        end = 1
    gap = (end - start) / 100000
    precisionVec_base = []
    recallVec_base = []
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprBase = 0.0
    recallTemp = 1.0

    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec_base.append(precision)
        recallVec_base.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision



    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprNew = 0.0
    recallTemp = 1.0
    precisionVec_our=[]
    recallVec_our=[]
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec_our.append(precision)
        recallVec_our.append(recall)
        auprNew += (recallTemp - recall) * precision
        recallTemp = recall
    auprNew += recall * precision

    # save graph
    plt.figure()
    plt.plot(recallVec_base,precisionVec_base, color="red", label="PR Curve baseline", lw=2)
    plt.plot(recallVec_our, precisionVec_our, color="blue", label="PR Curve Our", lw=2)
    #plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend()
    plt.savefig("./graph/PR_T_{}_M_{}.png".format(temperature, noise))
    return auprBase, auprNew


def auprOut(name):
    # calculate the AUPR
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 1
    if name == "CIFAR-100":
        start = 0.01
        end = 1
    gap = (end - start) / 100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprNew = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprNew += (recallTemp - recall) * precision
        recallTemp = recall
    auprNew += recall * precision
    return auprBase, auprNew


def detection(name):
    # calculate the minimum detection error
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 1
    if name == "CIFAR-100":
        start = 0.01
        end = 1
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    errorNew = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorNew = np.minimum(errorNew, (tpr + error2) / 2.0)

    return errorBase, errorNew


def metric(nn, data,temperture, noise):
    if nn == "densenet10" or nn == "wideresnet10": indis = "CIFAR-10"
    if nn == "densenet100" or nn == "wideresnet100": indis = "CIFAR-100"
    if nn == "densenet10" or nn == "densenet100": nnStructure = "VGG-BC-100"
    if nn == "wideresnet10" or nn == "wideresnet100": nnStructure = "Wide-ResNet-28-10"

    if data == "Imagenet_crop": dataName = "Tiny-ImageNet (crop)"
    if data == "Imagenet_resize": dataName = "Tiny-ImageNet (resize)"
    if data == "LSUN": dataName = "LSUN (crop)"
    if data == "LSUN_resize": dataName = "LSUN (resize)"
    if data == "iSUN": dataName = "iSUN"
    if data == "Gaussian": dataName = "Gaussian noise"
    if data == "Uniform": dataName = "Uniform Noise"
    if data == "CIFAR-100": dataName = "CIFAR-100"
    # CIFAR-10
    fprBase, fprNew = tpr95(indis)
    errorBase, errorNew = detection(indis)
    aurocBase, aurocNew = auroc(indis, temperture, noise)
    auprinBase, auprinNew = auprIn(indis, temperture, noise)
    auproutBase, auproutNew = auprOut(indis)
    print("{:31}{:>22}".format("Neural network architecture:", nnStructure))
    print("{:31}{:>22}".format("In-distribution dataset:", indis))
    print("{:31}{:>22}".format("Out-of-distribution dataset:", dataName))
    print("")
    print("{:>34}{:>19}".format("Baseline", "Our Method"))
    print("{:20}{:13.1f}%{:>18.1f}% ".format("FPR at TPR 95%:", fprBase * 100, fprNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("Detection error:", errorBase * 100, errorNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUROC:", aurocBase * 100, aurocNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR In:", auprinBase * 100, auprinNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR Out:", auproutBase * 100, auproutNew * 100))

    result=open("./softmax_scores/T_{}_noise_{}_d_{}.txt".format(temperture,noise,dataName), 'w')
    result.write("{:31}{:>22}\n".format("Neural network architecture:", nnStructure))
    result.write("{:31}{:>22}\n".format("In-distribution dataset:", indis))
    result.write("{:31}{:>22}\n".format("Out-of-distribution dataset:", dataName))
    result.write("{:>34}{:>19}\n".format("Baseline", "Our Method"))
    result.write("{:20}{:13.1f}%{:>18.1f}% \n".format("FPR at TPR 95%:", fprBase * 100, fprNew * 100))
    result.write("{:20}{:13.1f}%{:>18.1f}%\n".format("Detection error:", errorBase * 100, errorNew * 100))
    result.write("{:20}{:13.1f}%{:>18.1f}%\n".format("AUROC:", aurocBase * 100, aurocNew * 100))
    result.write("{:20}{:13.1f}%{:>18.1f}%\n".format("AUPR In:", auprinBase * 100, auprinNew * 100))
    result.write("{:20}{:13.1f}%{:>18.1f}%\n".format("AUPR Out:", auproutBase * 100, auproutNew * 100))











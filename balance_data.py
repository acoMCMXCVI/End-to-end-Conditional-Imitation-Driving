
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
import matplotlib.pyplot as plt

X = np.load('Data/X.npy', allow_pickle = True)
Y = np.load('Data/Y.npy', allow_pickle = True)


lefts = []
rights = []
forwards = []

for img, data in zip(X, Y):

    steer = data[1]

    if steer >= 0.02:
        rights.append([img,data])
    elif steer <= -0.02:
        lefts.append([img,data])
    else:
        forwards.append([img,data])


print("lefts: {} \nrights: {} \nforwards {}".format(len(lefts), len(rights), len(forwards)))

lefts = lefts[:len(rights)]
print("lefts: {} \nrights: {} \nforwards {}".format(len(lefts), len(rights), len(forwards)))

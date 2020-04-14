import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import cv2
import os
import sys


file_name = input('Name of train file:')

file_index = 0
X = []
Y = []

while True:

    file_name_index  = str(file_name) + '-' + str(file_index) + '.npy'

    if os.path.isfile(file_name_index):
        print('File ' + str(file_name_index) + ' exists, loading data!')
        training_data = np.load(file_name_index, allow_pickle=True)
    else:
        print('File ' + str(file_name_index) + ' does not exist!')

        file_name = input('Name of another train file:')
        file_index = 0
        file_name_index  = str(file_name) + '-' + str(file_index) + '.npy'

        if os.path.isfile(file_name_index):
            print('File ' + str(file_name_index) + ' exists, loading data!')
            training_data = np.load(file_name_index, allow_pickle=True)
        else:
            print('File ' + str(file_name_index) + ' does not exist!')
            break



    for data in training_data:

        image   =   data[0]
        control =   data[1]

        image = cv2.resize(image, (200,66)).reshape(-1,200,66,3)
        print(type(image))
        X.append(   image   )
        Y.append(   control )

    print(len(X))
    file_index += 1

print('Saving X...')
np.save('X1.npy',np.array(X))
print('Saving Y...')
np.save('Y1.npy',np.array(Y))

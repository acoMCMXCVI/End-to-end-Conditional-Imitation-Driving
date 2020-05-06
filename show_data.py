import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import cv2
import os
import sys



file_name = input('Name of train file:')
wkey = int(input('Wait key:'))

file_index = 0
lenght = 0

while True:

    file_name_index  = str(file_name) + '_' + str(file_index) + '.npy'

    if os.path.isfile(file_name_index):
        print('File ' + str(file_name_index) + ' exists, loading previous data!')
        training_data = np.load(file_name_index, allow_pickle=True)
    else:
        print('File ' + str(file_name_index) + ' does not exist!')
        break


    #lenght += len(training_data)
    print(len(training_data))

    for data in training_data:

        image = data[0]
        data_show = np.around(data[1], decimals=3)
        print(data_show)
        cv2.imshow('window', image)
        cv2.waitKey(wkey)

    file_index += 1



print(lenght)

#df = pd.DataFrame(training_data)
#print(df.head(20))
'''
while True:
    index = int(input('Enter your input:'))

    print(training_data[index][1])
    cv2.imshow('window', training_data[index][0])
    cv2.waitKey(0)


num_bins = 50
samples_per_bin = 200
hist, bins = np.histogram(df[1], num_bins)
center = bins[:-1] + bins[1:] * 0.5  # center the bins to 0

plt.bar(center, hist, width=0.05)
plt.plot((np.min(df[1]), np.max(df[1])), (samples_per_bin, samples_per_bin))

plt.show()

'''

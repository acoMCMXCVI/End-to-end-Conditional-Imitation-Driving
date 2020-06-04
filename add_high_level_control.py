import numpy as np
from random import shuffle
import keyboard

import cv2
import os
import sys

import time

from util import add_all_sesion_names

# adding all sesion to list
file_names = add_all_sesion_names()


data_files = 0
start = 0
end = 0

# making one big vector of hight_control
number_of_batches = input('Number of batches: ')

h_controls = np.zeros((1,int(number_of_batches)*256))
#cont = np.load('control.npy', allow_pickle=True)

print('O - starting point of high control.')
print('P - starting point of high control.')
print('1 - left, 2 - straight, 3 - right.')

time.sleep(2)



for file_name in file_names:

    print(file_name)
    file_index = 0


    done = False
    while not done:

        file_name_index  = 'Data/' + str(file_name) + '_' + str(file_index) + '.npy'

        if os.path.isfile(file_name_index):

            training_data = np.load(file_name_index, allow_pickle=True)
            #print(len(training_data))
            if len(training_data) > 256:
                training_data = training_data[:256][:]
                print('Training data was longer then 256.')

            for count, data in enumerate(training_data):

                image   =   data[0]
                control =   data[1][1]

                cv2.imshow('window', image)
                cv2.waitKey(8)

                # starting point of high control
                if keyboard.is_pressed("o"):
                    start = data_files
                    print('start frame is:' + str(start))
                # ending point of high control
                if keyboard.is_pressed('p'):
                    end = data_files
                    print('start frame is:' + str(start))
                    print('end frame is:' + str(end))

                    h_control = input('High control is:')

                    if h_control == '0' or h_control == '1' or h_control == '2' or h_control == '3':
                        print('Adding high control...')
                        h_controls[0][start:end] = h_control
                        start = end

                data_files += 1


        else:
            print('File dont exist!')
            done = True

        file_index += 1

# saving final high controls
np.save('Data/control.npy', h_controls)

import numpy as np
from random import shuffle
import keyboard

import cv2
import os
import sys


file_names = ['x', 'x1']


data_files = 0
start = 0
end = 0

h_controls = np.zeros((1,1113*257))
cont = np.load('control.npy', allow_pickle=True)
print(cont.shape)



for file_name in file_names:

    print(file_name)
    file_index = 0


    done = False
    while not done:

        file_name_index  = 'Data/' + str(file_name) + '_' + str(file_index) + '.npy'

        if os.path.isfile(file_name_index):

            training_data = np.load(file_name_index, allow_pickle=True)
            #print(len(training_data))

            for count, data in enumerate(training_data):


                image   =   data[0]
                control =   data[1][1]

                cv2.imshow('window', image)
                cv2.waitKey(8)

                if keyboard.is_pressed("o"):
                    start = data_files
                    print('start frame is:' + str(start))
                if keyboard.is_pressed('p'):
                    end = data_files
                    print('start frame is:' + str(start))
                    print('end frame is:' + str(end))

                    h_control = input('High control is:')

                    if h_control == '0' or h_control == '1' or h_control == '2' or h_control == '3':
                        print('Adding high control...')
                        h_controls[0][start:end] = h_control
                        start = end


                #showing data
                '''
                print(data_files)
                if cont[0][data_files] == 0:
                    print('Neutral')
                elif cont[0][data_files] == 1:
                    print('Left')
                elif cont[0][data_files] == 2:
                    print('Streith')
                else:
                    print('Right')
                '''


                data_files += 1

                #if data_files == 3000:
                #    np.save('control.npy', h_controls[0][0:3000])
                #    print('SACUVANO!!!')

        else:
            print('File dont exist!')
            done = True

        file_index += 1

#np.save('control.npy', h_controls)

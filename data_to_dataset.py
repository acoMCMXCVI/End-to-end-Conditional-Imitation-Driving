import numpy as np
from random import shuffle
import cv2
import os
import sys


h_control = np.load('Data/CONTROL.npy', allow_pickle=True)


file_names = ['x', 'x1']

img = []
control = []
Y = []

collect_set = []

lefts = []
rights = []
forwards = []

batch_index = 0
batch_X_img_name  = 'Data/Final/X_img_batch_' + str(batch_index) + '.npy'
batch_X_control_name  = 'Data/Final/X_control_batch_' + str(batch_index) + '.npy'
batch_Y_name  = 'Data/Final/Y_batch_' + str(batch_index) + '.npy'

data_files = 0
batch_num = 0

for file_name in file_names:

    print(file_name)
    file_index = 0

    done = False
    while not done:

        file_name_index  = 'Data/' + str(file_name) + '_' + str(file_index) + '.npy'

        if os.path.isfile(file_name_index):

            #print('File ' + str(file_name_index) + ' exists, loading data!')
            training_data = np.load(file_name_index, allow_pickle=True)
            h_control_batch = h_control[0][batch_num * 257 : (batch_num + 1) * 257]
            batch_num += 1

            #print(training_data.shape)
            if len(training_data) > 257:
                training_data = training_data[:257][:]

            for count, data in enumerate(training_data):
                #print('count:{}'.format(count))
                if data[1][1] >= 0.02:
                    image   =   data[0][-66:,:,:]
                    control =   data[1][1]
                    rights.append([image, h_control_batch[count], control])
                    #print('idemo')
                elif data[1][1] <= -0.02:
                    image   =   data[0][-66:,:,:]
                    control =   data[1][1]
                    lefts.append([image, h_control_batch[count], control])
                else:
                    image   =   data[0][-66:,:,:]
                    control =   data[1][1]
                    forwards.append([image, h_control_batch[count], control])

                if (len(rights) + len(lefts) + len(forwards)) % 2500 == 0:
                    print((len(rights) + len(lefts) + len(forwards)))
                #image   =   data[0][-66:,:,:]
                #control =   data[1]

                #collect_set.append ([image, control])


                #cv2.imshow('window', image)
                #cv2.waitKey(2)


            if (len(rights) + len(lefts) + len(forwards)) > 10000:
                print('File ' + str(file_name_index) + ' exists, loading data!')

                print('Shuffleing...')
                shuffle(forwards)
                shuffle(lefts)
                shuffle(rights)

                print('Shuffle done!')

                print('Lefts: ' + str(len(lefts)) + '\tForwards: ' + str(len(forwards)) + '\tRights: ' + str(len(rights)))

                forwards = forwards[:len(lefts)][:len(rights)]
                lefts = lefts[:len(forwards)]
                rights = rights[:len(forwards)]
                print('Slice done!')

                final_data = forwards + lefts + rights
                shuffle(final_data)


                lefts = []
                rights = []
                forwards = []

                print ('Saving: ' + str(len(final_data)) + 'files!')

                num_of_batches = int(len(final_data) / 256)
                for k in range(0, num_of_batches):

                    batch = final_data[k * 256 : (k + 1) * 256 ]
                    img = []
                    control = []
                    Y = []

                    for i in batch:
                        img.append(i[0])
                        control.append(i[1])
                        Y.append(i[2])

                    np.save(batch_X_img_name, img)
                    np.save(batch_X_control_name, control)
                    np.save(batch_Y_name, Y)

                    print(str(batch_X_img_name) + 'is saved!')


                    batch_index += 1
                    batch_X_img_name  = 'Data/Final/X_img_batch_' + str(batch_index) + '.npy'
                    batch_X_control_name  = 'Data/Final/X_control_batch_' + str(batch_index) + '.npy'
                    batch_Y_name  = 'Data/Final/Y_batch_' + str(batch_index) + '.npy'

                data_files += len(final_data)
                final_data = []


                print('Ukupno sacuvano: ' + str(data_files))

            #print ('lefts: {} \tforwards: {} \trights: {}'.format(lefts, forwards, rights))
            #lefts: 11727    forwards: 258740        rights: 15323 => 0.2
            #lefts: 16328    forwards: 250031        rights: 19431 => 0.15
            #lefts: 27057    forwards: 228817        rights: 29916 => 0.1
            #lefts: 52858    forwards: 178756        rights: 54176 => 0.05
            #lefts: 78932    forwards: 122780        rights: 84078 => 0.02
            #lefts: 79094    forwards: 122783        rights: 83915
            # bilo 194k


        else:
            print('File dont exist!')
            done = True


        #print(batch_num)
        file_index += 1

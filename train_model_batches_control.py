import numpy as np
from functional_control_end_to_end_keras_model import nvidia_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from time import time
import os




def load_batches():

    batch_index = 0

    while True:
        batch_X_img_name  = 'Data/Batches_HIGH_CONTROL/X_img_batch_' + str(batch_index) + '.npy'
        batch_X_control_name  = 'Data/Batches_HIGH_CONTROL/X_control_batch_' + str(batch_index) + '.npy'
        batch_Y_name  = 'Data/Batches_HIGH_CONTROL/Y_batch_' + str(batch_index) + '.npy'

        X_img = np.load(batch_X_img_name, allow_pickle=True)
        X_control = np.load(batch_X_control_name, allow_pickle=True)
        Y = np.load(batch_Y_name, allow_pickle=True)

        one_hot = np.zeros((X_control.size, X_control.astype(int).max()+1))
        one_hot[np.arange(X_control.size), X_control.astype(int)] = 1


        #print(one_hot.shape)
        #print(X_img.shape)


        batch_index += 1

        yield ([X_img, one_hot], Y)

        if batch_index == 744:
            batch_index = 0




width = 200
height = 66
depth = 3
lr = 1e-3
epochs = 12


model_name = 'Models/carla-{}-{}-{}-epochs-batches.h5'.format(lr, 'nvidiaETE', epochs)


model = nvidia_model(width, height, depth, 4)

model.compile(optimizer = Adam(lr=lr), loss='mse')


checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='auto')


logdir = os.path.join("Log",str(time()),)
tensorboard = TensorBoard(log_dir=logdir)


history_obj = model.fit_generator(load_batches(), steps_per_epoch=744, epochs=epochs, callbacks = [checkpointer, tensorboard], verbose=1)

#history_obj = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_data = (X_val, y_val), callbacks = [checkpointer, tensorboard], shuffle=True, verbose=1)

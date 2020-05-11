import numpy as np
from end_to_end import nvidia_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from time import time
import os

#batch_index = 0

#batch_X_name  = 'Data/Final/X_batch_' + str(batch_index) + '.npy'
#batch_Y_name  = 'Data/Final/Y_batch_' + str(batch_index) + '.npy'


def load_batches():

    batch_index = 0

    while True:
        batch_X_name  = 'Data/Final/X_batch_' + str(batch_index) + '.npy'
        batch_Y_name  = 'Data/Final/Y_batch_' + str(batch_index) + '.npy'

        X = np.load(batch_X_name, allow_pickle=True)
        Y = np.load(batch_Y_name, allow_pickle=True)

        batch_index += 1

        yield (X,Y)

        if batch_index == 744:
            batch_index = 0




width = 200
height = 66
depth = 3
lr = 1e-3
epochs = 7


model_name = 'Models/carla-{}-{}-{}-epochs-batches.h5'.format(lr, 'nvidiaETE', epochs)




model = nvidia_model(width, height, depth)

model.compile(optimizer = Adam(lr=lr), loss='mse')



checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='auto')


logdir = os.path.join("Log",str(time()),)
tensorboard = TensorBoard(log_dir=logdir)

history_obj = model.fit_generator(load_batches(), steps_per_epoch=744, epochs=epochs, callbacks = [checkpointer, tensorboard], verbose=1)

#history_obj = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_data = (X_val, y_val), callbacks = [checkpointer, tensorboard], shuffle=True, verbose=1)

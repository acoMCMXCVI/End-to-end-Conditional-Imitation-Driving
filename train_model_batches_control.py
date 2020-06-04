import numpy as np
from functional_conditional_end_to_end_keras_model import nvidia_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split

from time import time
import os

from util import image_augmentation

seq = image_augmentation()

def load_batches():

    batch_index = 0

    while True:
        batch_X_img_name  = 'Data/Final/X_img_batch_' + str(batch_index) + '.npy'
        batch_X_control_name  = 'Data/Final/X_control_batch_' + str(batch_index) + '.npy'
        batch_Y_name  = 'Data/Final/Y_batch_' + str(batch_index) + '.npy'

        X_img = np.load(batch_X_img_name, allow_pickle=True)
        X_control = np.load(batch_X_control_name, allow_pickle=True)
        Y = np.load(batch_Y_name, allow_pickle=True)

        # making one hot array from high-level control
        one_hot_control = np.zeros((X_control.size, X_control.astype(int).max()+1))
        one_hot_control[np.arange(X_control.size), X_control.astype(int)] = 1

        # data augmentation
        aug_X_img = seq(images=X_img)


        batch_index += 1

        yield ([aug_X_img, one_hot_control], Y)

        if batch_index == 744:
            batch_index = 0




width = 200
height = 66
depth = 3
lr = 1e-4
epochs = 4


model_name = 'Models/carla-{}-{}-{}-epochs-batches-aug-2nd.h5'.format(lr, 'nvidiaETE', epochs)


model = nvidia_model(width, height, depth, 4)

model.compile(optimizer = Adam(lr=lr), loss='mse')

# continue training
#model = load_model(model_name)

checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='auto')

logdir = os.path.join("Log",str(time()),)
tensorboard = TensorBoard(log_dir=logdir)


history_obj = model.fit_generator(load_batches(), steps_per_epoch=744, epochs=epochs, callbacks = [checkpointer, tensorboard], verbose=1)

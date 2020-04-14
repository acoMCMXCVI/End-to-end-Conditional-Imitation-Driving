import numpy as np
from end_to_end import nvidia_model
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from time import time
import os


width = 200
height = 66
lr = 1e-3
epochs = 10
depth = 3


X = np.load('Data/X.npy', allow_pickle = True)
Y = np.load('Data/Y.npy', allow_pickle = True)

print(X.shape)

model_name = 'Models/carla-{}-{}-{}-epochs.h5'.format(lr, 'nvidiaETE', epochs)

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.33)

X_val = X_val.reshape(-1,width,height,3)
X_train = X_train.reshape(-1,width,height,3)


print(X_train.shape)

model = nvidia_model(width, height, depth)

model.compile(optimizer = Adam(lr=lr), loss='mse')



checkpointer = ModelCheckpoint(model_name, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')


logdir = os.path.join("Log",str(time()),)
tensorboard = TensorBoard(log_dir=logdir)

history_obj = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_data = (X_val, y_val), callbacks = [checkpointer, tensorboard], shuffle=True, verbose=1)

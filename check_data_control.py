import numpy as np
import cv2
import keyboard

image = np.load('X_img_batch_600.npy', allow_pickle=True)
control = np.load('X_control_batch_600.npy', allow_pickle=True)
y = np.load('Y_batch_600.npy', allow_pickle=True)

index = 0


while True:
    print(y[index])
    print(control[index])
    cv2.imshow('window', image[index])
    cv2.waitKey(0)

    if keyboard.is_pressed('a'):
        index += 1
    elif keyboard.is_pressed('d'):
        index -= 1
    elif keyboard.is_pressed('q'):
        break

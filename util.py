import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

def shuffle_data(X_train, Y_train):

    m = len(X_train)

    permutation = list(np.random.permutation(m))
    shuffled_X = X_train[:, permutation]
    shuffled_Y = Y_train[:, permutation]

    return shuffled_X, shuffled_Y


def image_augmentation():

    seq = iaa.SomeOf((0, None),[

        iaa.GaussianBlur( sigma=(0, 0.8) ),

        iaa.AdditiveGaussianNoise( scale=(0, 0.05*255) ),

        iaa.SaltAndPepper((0.005, 0.03)),

        iaa.CoarseDropout( (0.0, 0.05), size_percent = (0.02, 0.25) ),

        iaa.LinearContrast((0.4, 1.6)),

        #iaa.ChangeColorTemperature((4100, 9000)),

        iaa.MultiplyAndAddToBrightness(mul=1, add=(-80, 80)),


    ], random_order=True)

    return seq


def add_all_sesion_names():
    # adding all sesion to list
    file_names = []

    ses = True
    while ses:
        # adding all sesion to list
        name_of_sesion = input('Name_of_colecting_sesion: ')
        file_names.append(name_of_sesion)

        moreSes = input('Is there more sesions? (Y / N): ')
        if moreSes == 'N':
            ses = False

    print('All sesions names: {}'.format(file_names))

    return file_names

import numpy as np

def shuffle_data(X_train, Y_train):

    m = len(X_train)

    permutation = list(np.random.permutation(m))
    shuffled_X = X_train[:, permutation]
    shuffled_Y = Y_train[:, permutation]

    return shuffled_X, shuffled_Y

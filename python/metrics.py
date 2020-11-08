import numpy as np

def MSE(y, y_tilde):
    """MSE"""
    return np.mean(np.square(y - y_tilde), axis=0)

def R_2(y, y_tilde):
    """R2"""
    return 1 - np.sum(np.square(y - y_tilde)) / np.sum(np.square(y - np.mean(y)))

model_variance = lambda y_tilde: np.mean((y_tilde - np.mean(y_tilde, axis=0)) ** 2)

model_bias = lambda y, y_tilde: np.mean((y - np.mean(y_tilde, axis=0)) ** 2)

def cross_entropy(y, y_tilde):
    """Cross entropy"""
    return - np.mean(y * np.log(y_tilde) + (1 - y) * np.log(1 - y_tilde))

#cross_entropy = lambda y, y_tilde: np.mean(y * np.log(y_tilde) + (1 - y) * np.log(1 - y_tilde))

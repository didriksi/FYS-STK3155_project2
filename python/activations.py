import numpy as np

def sigmoid(z):
    """$\\sigma$"""
    return 1/(1 + np.exp(-z))

def sigmoid_diff(z):
    """$\\sigma'$"""
    a = sigmoid(z)
    return a*(1-a)

def ReLu(z):
    """ReLu"""
    return np.where(z > 0, z, 0)

def ReLu_diff(z):
    """ReLu'"""
    return np.where(z > 0, 1, 0)

def leaky_ReLu(z):
    """leaky ReLu"""
    return np.where(z > 0, z, z * 0.01) 

def leaky_ReLu_diff(z):
    """(leaky ReLu)'"""
    return np.where(z > 0, 1, 0.01)

def linear(z):
    """identity"""
    return z

def linear_diff(z):
    """identity'"""
    return np.ones(z.shape)

def softmax(z):
    """softmax"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def softmax_diff(z):
    """softmax'"""
    grad = np.diag(z) - np.dot(z, z.T)
    return grad

sigmoids = (sigmoid, sigmoid_diff)
relus = (ReLu, ReLu_diff)
leaky_relus = (leaky_ReLu, leaky_ReLu_diff)
linears = (linear, linear_diff)
softmaxs = (softmax, softmax_diff)
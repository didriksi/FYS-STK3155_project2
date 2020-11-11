import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_diff(z):
    a = sigmoid(z)
    return a*(1-a)

def ReLu(z):
    return np.where(z > 0, z, 0)

def ReLu_diff(z):
    return np.where(z > 0, 1, 0)

def leaky_ReLu(z):
    return np.where(z > 0, z, z * 0.01) 

def leaky_ReLu_diff(z):
    return np.where(z > 0, 1, 0.01)

def linear(z):
    return z

def linear_diff(z):
    return np.ones(z.shape)

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def softmax_diff(z):
    grad = np.diag(z) - np.dot(z, z.T)
    return grad

sigmoids = (sigmoid, sigmoid_diff)
relus = (ReLu, ReLu_diff)
leaky_relus = (leaky_ReLu, leaky_ReLu_diff)
linears = (linear, linear_diff)
softmaxs = (softmax, softmax_diff)
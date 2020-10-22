import numpy as np

MSE = lambda y, y_tilde: np.mean(np.square(y - y_tilde), axis=0)

R_2 = lambda y, y_tilde: 1 - np.sum(np.square(y - y_tilde))/np.sum(np.square(y - np.mean(y)))

model_variance = lambda y_tilde: np.mean((y_tilde - np.mean(y_tilde, axis=0))**2)

model_bias = lambda y, y_tilde: np.mean((y - np.mean(y_tilde, axis=0))**2)
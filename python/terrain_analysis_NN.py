import numpy as np
import matplotlib.pyplot as plt

from sgd import sgd, plot_sgd
import neural_model as nn
import real_terrain


data = real_terrain.get_data(20)
polynomials = 8
x_train = data['x_train']
y_train = data['y_train'][:, np.newaxis]
inputs = int(x_train.shape[1])


Neural_net = nn.Network()
Neural_net.addLayer(nn.Input(inputs))
Neural_net.addLayer(nn.Dense(10))
Neural_net.addLayer(nn.Output(1, activation=nn.linear, diff_activation=nn.linear_diff))
Neural_net.compile()
print(Neural_net)
learning_rates = [0.06, 0.1, 2]

learning_rate_iter = [(learning_rate, f"flat_{learning_rate}") for learning_rate in np.logspace(-3, -1, 3)]

errors = sgd(x_train, y_train, learning_rate_iter, model=Neural_net, epochs=50, mini_batch_size=10)

plt.plot(errors)
plt.show()

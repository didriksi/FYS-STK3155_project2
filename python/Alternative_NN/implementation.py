import matplotlib.pyplot as plt
import numpy as np

import mnist20
import net

train_data, val_data, test_data = mnist20.load_data_wrapper()

NN = net.Network([64,32,10])

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = NN.SGD(train_data, 100, 10, 0.3,
                                                                                evaluation_data=val_data,
                                                                                monitor_training_accuracy=True)
plt.plot(training_accuracy)
plt.show()

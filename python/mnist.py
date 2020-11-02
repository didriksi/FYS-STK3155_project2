import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import data_handling
import plotting

def get_data(train_size, test_size):
    """Returns MNIST data dict with training, validation and testing data randomly.

    The MNIST dataset is handdrawn digits.

    Parameters:
    -----------
    train_size: float
                Number between 0 and 1 that gives the portion of data that should be training data.
                Sum of train_size and test_size must be no larger than 1.

    test_size:  float
                Number between 0 and 1 that gives the portion of data that should be testing data.
                If the sum of train_size and test_size is under 1, the rest of the data is labeled as validation data.
    
    Returns:
    --------
    data:       dict
                Big dictionary with training, testing and validation data.
    """

    assert train_size + test_size <= 1, "Sum of train_size and test_size must be no larger than 1."

    mnist_x, mnist_y = load_digits(return_X_y=True)

    data = {}

    data['x_train_validate'], data['x_test'], data['y_train_validate'], data['y_test'] = train_test_split(mnist_x, mnist_y, test_size=test_size)

    data['x_train'], data['x_validate'], data['y_train'], data['y_validate'] = train_test_split(data['x_train_validate'], data['y_train_validate'], test_size=(1 - test_size - train_size))

    data['x'] = np.concatenate((data['x_train'], data['x_validate'], data['x_test']), axis=0)
    data['y'] = np.concatenate((data['y_train'], data['y_validate'], data['y_test']), axis=0)

    return data

if __name__ == '__main__':
    get_data(0.6, 0.2)
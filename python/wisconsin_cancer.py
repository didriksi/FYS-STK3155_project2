import sklearn.datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def onehot(y, max_y=None):
    """One hot encoder for 1d arrays where each element is a data point

    Paramaters:
        y:      1darray
                Array of output data eg. [1. 2. 4. 1. 0. 5.]
        max_y   int
                number of classes in y
    """
    if max_y==None:
        max_y = np.argmax(y)
    vector = np.zeros((y.shape[0], max_y))
    ints = np.arange(y.shape[0])
    vector[ints, y] = 1
    return vector

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

    x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    y = onehot(y, 2)

    data = {}

    data['x_train_validate'], data['x_test'], data['y_train_validate'], data['y_test'] = train_test_split(
        x, y, test_size=test_size)
    test_size = 1 - test_size - train_size
    data['x_train'], data['x_validate'], data['y_train'], data['y_validate'] = train_test_split(
            data['x_train_validate'], data['y_train_validate'], test_size=test_size)

    data['x'] = np.concatenate((data['x_train'], data['x_validate'], data['x_test']), axis=0)
    data['y'] = np.concatenate((data['y_train'], data['y_validate'], data['y_test']), axis=0)

    return data

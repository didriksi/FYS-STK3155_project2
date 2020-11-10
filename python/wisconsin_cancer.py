import sklearn.datasets
import numpy as np

import data_handling

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
    """Returns cancer data dict with training, validation and testing data randomly.

    The dataset is known as the Wisconcin Cancer dataset.

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
    x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

    data = data_handling.make_classification_dict(x, y, train_size, test_size, even=True)
    
    return data

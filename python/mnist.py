import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import data_handling
import plotting

def onehot(y, max_y):
    vector = np.zeros((y.shape[0], max_y))
    vector[:,y] = 1
    return vector


def get_data(train_size, test_size, even=True, truncate=False):
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
    even:       bool. default=True
                Even out distributions between different classes in the training and validation sets.
    truncate:   bool. default=False
                Truncates the dataset so each class has as many datapoints. Not implemented.

    Returns:
    --------
    data:       dict
                Big dictionary with training, testing and validation data.
    """

    assert train_size + test_size <= 1, "Sum of train_size and test_size must be no larger than 1."
    
    if truncate:
        raise NotImplementedError("Haven't implemented option to truncate yet.")

    mnist_x, mnist_y = load_digits(return_X_y=True)
    mnist_x /= np.max(mnist_x)

    data = {}

    if even:
        class_length = [sum(number_class) for number_class in [mnist_y==y_ for y_ in range(10)]]

        training_validation_samples = int(np.mean(class_length)*(1 - test_size))
        training_samples = int(np.mean(class_length)*(train_size))
        testing_samples = int(np.mean(class_length)*(test_size))

        x_train = [None]*10
        x_validate = [None]*10
        x_test = [None]*10
        
        y_train = [None]*10
        y_validate = [None]*10
        y_test = [None]*10

        for y_ in range(10):
            class_samples = np.nonzero(mnist_y==y_)[0]
            indices = np.random.permutation(class_samples)
            train = indices[:training_samples]
            validate = indices[training_samples:training_validation_samples]
            test = indices[training_validation_samples:]

            x_train[y_] = mnist_x[train]
            y_train[y_] = onehot(mnist_y[train], 10)
            x_validate[y_] = mnist_x[validate]
            y_validate[y_] = onehot(mnist_y[validate], 10)
            x_test[y_] = mnist_x[test]
            y_test[y_] = onehot(mnist_y[test], 10)

        shuffled_training_indexes = np.random.permutation(training_samples*10)
        shuffled_validation_indexes = np.random.permutation((training_validation_samples-training_samples)*10)
        shuffled_test_indexes = np.random.permutation(testing_samples*10)

        data['x_train'] = np.concatenate(x_train)[shuffled_training_indexes]
        data['x_validate'] = np.concatenate(x_validate)[shuffled_validation_indexes]
        data['x_test'] = np.concatenate(x_test)[shuffled_test_indexes]
        
        data['y_train'] = np.concatenate(y_train)[shuffled_training_indexes]
        data['y_validate'] = np.concatenate(y_validate)[shuffled_validation_indexes]
        data['y_test'] = np.concatenate(y_test)[shuffled_test_indexes]

    else:
        data['x_train_validate'], data['x_test'], data['y_train_validate'], data['y_test'] = train_test_split(mnist_x, mnist_y, test_size=test_size)

        data['x_train'], data['x_validate'], data['y_train'], data['y_validate'] = train_test_split(data['x_train_validate'], data['y_train_validate'], test_size=(1 - test_size - train_size))

    data['x'] = np.concatenate((data['x_train'], data['x_validate'], data['x_test']), axis=0)
    data['y'] = np.concatenate((data['y_train'], data['y_validate'], data['y_test']), axis=0)
    return data

def plot_data(*data_args):
    """Plots data from the get_data function.

    Parameters:
    -----------
    *data_args: args
                Passed on to get_data
    """

    data = get_data(*data_args)

    # Plot how many data points there are of each class in each subset
    plt.hist([data['y_train'], data['y_validate'], data['y_test']], stacked=True, rwidth=0.9, bins=np.arange(0,11)-0.5)
    plt.xticks(range(10))
    plt.legend(["Training", "Validation", "Testing"])
    plt.savefig("../plots/mnist_classes.png")
    plt.close()

    # Plot sum of pixel values for each bin and each dataset
    for dataset in ["train", "validate", "test"]:
        x = data[f'x_{dataset}']
        y = data[f'y_{dataset}']

        xs = np.empty(10)
        for y_ in range(10):
            xs[y_] = np.sum(x[y==y_])

        plt.plot(xs/np.sum(xs))
    plt.ylim(0,0.2)
    plt.xticks(range(10))
    plt.legend(["Training", "Validation", "Testing"])
    plt.savefig("../plots/mnist_sum_of_pixels_per_class.png")
    plt.close()

    # Plot some sample images
    fig = plt.figure()
    for y_ in range(10):
        ax = fig.add_subplot(2, 5, y_+1)
        ax.imshow(data['x'][data['y']==y_][0].reshape(8,8), cmap=plt.get_cmap('Greys'))
        ax.set_title(y_)
        plt.xticks([])
        plt.yticks([])
    plt.savefig("../plots/mnist_samples.png")

if __name__ == '__main__':
    plot_data(0.6, 0.2)
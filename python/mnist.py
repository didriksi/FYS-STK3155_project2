import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import data_handling
import plotting

def get_data(train_size, test_size, even=True):
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

    Returns:
    --------
    data:       dict
                Big dictionary with training, testing and validation data.
    """

    mnist_x, mnist_y = load_digits(return_X_y=True)

    return data_handling.make_classification_dict(mnist_x, mnist_y, train_size, test_size, even=even)

def plot_data(data):
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
    data = get_data(0.6, 0.2)
    plot_data(data)
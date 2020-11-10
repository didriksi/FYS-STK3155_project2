import numpy as np
from sklearn.model_selection import train_test_split

def make_data_dict(y, x_sparsity):
    """Takes in grid-size for training data, and predictor data, and makes big dict with training, validation and testing data.

    Is memory intensive for large datasets, but precomputes all possible sets and masks. Unsure of how big the dataset has
    to be, or how much the datasets have to be used before it's efficient.

    Parameters:
    -----------
    y:          array of shape (n, n)
                Dependent variable based on some two dimensional predictor variable.
    x_sparsity: int
                Describing grid-size for training and validation data.
    Returns:
    --------
    data:       dict
                Big dictionary with training, testing and validation data, as well as masks.
    """

    data = {}
    data['y'] = y[:y.shape[0]//x_sparsity*x_sparsity,:y.shape[1]//x_sparsity*x_sparsity].astype(float)/400   

    # Training data
    x1, x2 = np.meshgrid(np.arange(0, data['y'].shape[0], x_sparsity, dtype=int), np.arange(0, data['y'].shape[1], x_sparsity, dtype=int))
    data['x_train'] = np.array([x1, x2]).T.reshape(-1, 2)
    data['train_m'] = np.s_[:data['x_train'].shape[0]]
    data['y_train'] = data['y'][data['x_train'][:,0],data['x_train'][:,1]][:,np.newaxis]

    # Validation data
    x1, x2 = np.meshgrid(np.arange(x_sparsity//2, data['y'].shape[0], x_sparsity, dtype=int), np.arange(x_sparsity//2, data['y'].shape[1], x_sparsity, dtype=int))
    data['x_validate'] = np.array([x1, x2]).T.reshape(-1, 2)
    data['validate_m'] = np.s_[data['x_train'].shape[0]:data['x_train'].shape[0]+data['x_validate'].shape[0]]
    data['y_validate'] = data['y'][data['x_validate'][:,0],data['x_validate'][:,1]][:,np.newaxis]  

    # Complete x
    x1, x2 = np.meshgrid(np.arange(0, data['y'].shape[0], dtype=int), np.arange(0, data['y'].shape[1], dtype=int))
    data['x'] = np.array([x1, x2]).T.reshape(-1, 2)

    # Test data
    test = np.ones(data['y'].shape, dtype=bool)
    test[data['x_train'][:,0],data['x_train'][:,1]] = 0
    test[data['x_validate'][:,0],data['x_validate'][:,1]] = 0
    return_counts_for_complete_and_mask = np.unique(np.append(data['x'], np.array(np.nonzero(test)).T, axis=0), axis=0, return_counts=True)[1]
    data['test_m'] = np.s_[np.array([return_counts_for_complete_and_mask[:data['x'].shape[0]] - 1], dtype=bool).T[:,0]]
    data['x_test'] = data['x'][data['test_m']]
    data['y_test'] = data['y'][data['x_test'][:,0],data['x_test'][:,1]][:,np.newaxis]

    data['x_train_validate'] = np.append(data['x_train'], data['x_validate'], axis=0)
    data['y_train_validate'] = data['y'][data['x_train_validate'][:,0],data['x_train_validate'][:,1]][:,np.newaxis]

    data['x_train'] = data['x_train'].astype(float)/400
    data['x_train_validate'] = data['x_train_validate'].astype(float)/400
    data['x_validate'] = data['x_validate'].astype(float)/400
    data['x_test'] = data['x_test'].astype(float)/400
    return data

def make_classification_dict(x, y, train_size, test_size, even=True):
    """Makes a data dict with training, testing and validation data, randomly.

    Parameters
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
    assert train_size + test_size <= 1, "Sum of train_size and test_size must be no larger than 1."

    x /= np.max(x)

    data = {}

    if even:
        class_length = [sum(number_class) for number_class in [y==y_ for y_ in range(10)]]

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
            class_samples = np.nonzero(y==y_)[0]
            indices = np.random.permutation(class_samples)
            train = indices[:training_samples]
            validate = indices[training_samples:training_validation_samples]
            test = indices[training_validation_samples:]

            x_train[y_] = x[train]
            y_train[y_] = onehot(y[train], 10)
            x_validate[y_] = x[validate]
            y_validate[y_] = onehot(y[validate], 10)
            x_test[y_] = x[test]
            y_test[y_] = onehot(y[test], 10)

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

def onehot(y, max_y):
    vector = np.zeros((y.shape[0], max_y))
    vector[:,y] = 1
    return vector
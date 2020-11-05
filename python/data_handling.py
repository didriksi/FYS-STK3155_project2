import numpy as np
from sklearn.model_selection import train_test_split

def make_data_dict(y, *selection_parameter, random=False):
    """Takes in grid-size for training data, and predictor data, and makes big dict with training, validation and testing data.

    Is memory intensive for large datasets, but precomputes all possible sets and masks. Unsure of how big the dataset has
    to be, or how much the datasets have to be used before it's efficient.

    Parameters:
    -----------
    y:          array of shape (n, n)
                Dependent variable based on some two dimensional predictor variable.
    selection_parameter:
                
                If random == True, this is two floats between 0 and 1, how much of the data should be training data,
                and how much should be validation data.
                If random == False, this is x_sparsity, describing grid-size for training and validation data.
    random:     bool
                Whether or not to pick training, validation and testing data at random.
                Leave at true if you want to pick from a grid.
    Returns:
    --------
    data:       dict
                Big dictionary with training, testing and validation data, as well as masks.
    """

    if random:
        data = {}
        x1, x2 = np.meshgrid(np.arange(0, y.shape[0], dtype=int), np.arange(0, y.shape[1], dtype=int))
        x = np.array([x1, x2]).T.reshape(-1, 2)

        data['x_train_validate'], _, data['x_test'], _ = train_test_split(x, x, test_size=1-np.sum(selection_parameter))
        data['y_train_and_validate'] = y[data['x_train_validate'][:,0], data['x_train_validate'][:,1]]
        data['y_test'] = y[data['x_test'][:,0], data['x_test'][:,1]]

        data['x_train'], _, data['x_validate'], _ = train_test_split(data['x_train_validate'], data['x_train_validate'], test_size=selection_parameter[1])
        data['y_train'] = y[data['x_train'][:,0], data['x_train'][:,1]]
        data['y_validate'] = y[data['x_validate'][:,0], data['x_validate'][:,1]]

        data['x'] = np.append(data['x_train_validate'], data['x_test'], axis=0)
        data['y'] = y

    else:
        x_sparsity = selection_parameter[0]
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tune
import metrics
import linear_models

def sgd(data, mini_batches, epochs, learning_rate_iter, polynomials, momentum=0, _lambda=0, initial_conditions=1, metric=metrics.MSE):
    """Stochastic gradient descent

    Parameters:
    -----------
    data:       dict
                All training and testing data, as constructed by data_handling.make_data_dict.
    mini_batches:
                int
                Number of mini-batches.
    epochs:     int
                Number of epochs.
    learning_rate_iter:
                iterable (int -> float)
                Iterable object of functions that takes in a timestep, and returns a learning rate.
    polynomials:int
                Maximum degree of polynomials in design matrix
    momentum:   float
                Gamma parameter, determining how much momentum to keep. Set to zero for standard SGD.
    _lambda:    float
                Shrinkage parameter for Ridge regression. Set to 0 for OLS.
    initial_conditions:
                int
                Amount of attempts with different random initial conditions.
    metric:     array of shape(n, ), array of shape(n, ) -> float
                Should take in y and y_tilde, and return the total error.

    
    Returns:
    --------
    errors:     pd.DataFrame
                Dataframe with initial conditions and learning rates as multiindex, and epochs as column
    """

    errors_index = pd.MultiIndex.from_product([
        range(initial_conditions),
        range(len(learning_rate_iter))],
        names = ['Initial condition number', 'Learning rate number']
        )
    errors = pd.DataFrame(dtype=float, index=errors_index, columns=range(1, epochs + 1))
    errors.sort_index(inplace=True)
    print(errors)

    data_size = data['x_train'].shape[0]
    assert data_size%mini_batches == 0, "Data must be divisible with mini_batches"

    step = 0
    X = tune.poly_design_matrix(polynomials, data['x_train'])

    init_betas = np.random.randn(initial_conditions, X.shape[1])
    for i, beta in enumerate(init_betas): # TODO: Vectorise this loop away
        for j, learning_rate in enumerate(learning_rate_iters): # TODO: Vectorise this loop away
            for epoch in range(1, epochs + 1):
                print('epoch: ', epoch)
                v = np.zeros_like(beta)
                indexes_used = np.zeros(data_size, dtype=bool)
                for mini_batch in range(1, mini_batches + 1):
                    k = np.random.choice(np.arange(data_size)[indexes_used == False], size=(int(data_size/mini_batches)))
                    indexes_used[k] = True

                    y_tilde = X[k] @ beta

                    # TODO: Implement classes in linear_models.py in a way where this can come from it
                    cost_diff = np.dot(X[k].T, 2/(data_size/mini_batches)*(y_tilde - data['y_train'][k])) + 2*_lambda*beta

                    v = momentum*v + learning_rate(step)*cost_diff
                    beta = beta - v

                    step += 1

                errors.loc[i,j][epoch] = metrics.MSE(X @ beta, data['y_train'])

    return errors

if __name__ == '__main__':
    import real_terrain

    data = real_terrain.get_data(20)

    epochs = 20
    X_test = tune.poly_design_matrix(6, data['x_train'])

    # TODO: Make class or some other datastructure for learning rate functions, so that it has a name?
    learning_rate_iters = [lambda t: learning_rate for learning_rate in np.logspace(-4, -1, 4)]
    errors = sgd(data, 20, epochs, learning_rate_iters, 6, momentum=0.5, _lambda=0.01, initial_conditions=5)

    for j in range(len(learning_rate_iters)):
        error_for_learning_rate = errors.xs(j, level=1, drop_level=False)
        min_errors = error_for_learning_rate.min(axis=0)
        max_errors = error_for_learning_rate.max(axis=0)
        best_init_condition = error_for_learning_rate[epochs].idxmin()
        best_errors = error_for_learning_rate.loc[best_init_condition]
        # TODO: Make plot with min_errors and max_errors as extremes, and best_errors as center.
    
    model = linear_models.LinearRegression()
    model.fit(X_test, data['y_train'])
    plt.plot([0, epochs], np.repeat(metrics.MSE(model.predict(X_test), data['y_train']), 2), label="Analytical")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.legend()
    plt.savefig("../plots/sgd_different_learning_rates")
    #plt.show()



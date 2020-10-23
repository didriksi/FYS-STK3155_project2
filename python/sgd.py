import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tune
import metrics
import linear_models
import plotting

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
                iterable ((int -> float), str)
                Iterable object of a tuple of functions that takes in a timestep, and returns a learning rate,
                and a str indication name of learning rate function.
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
        [learning_rate_name for _, learning_rate_name in learning_rate_iter],
        range(initial_conditions)],
        names = ['Learning rate number', 'Initial condition number']
        )
    errors = pd.DataFrame(dtype=float, index=errors_index, columns=range(1, epochs + 1))
    errors.sort_index(inplace=True)
    print(errors)

    data_size = data['x_train'].shape[0]
    assert data_size%mini_batches == 0, "Data must be divisible with mini_batches"

    step = 0
    X = tune.poly_design_matrix(polynomials, data['x_train'])

    beta = np.random.randn(X.shape[1], initial_conditions)
    # TODO: Vectorise this loop away. --Seems difficult as long as it's a function that might include other factors than step
    for learning_rate_func, learning_rate_name in learning_rate_iter:
        for epoch in range(1, epochs + 1):
            print('epoch: ', epoch)
            v = np.zeros_like(beta)
            indexes_used = np.zeros(data_size, dtype=bool)
            for mini_batch in range(1, mini_batches + 1):
                k = np.random.choice(np.arange(data_size)[indexes_used == False], size=(int(data_size/mini_batches)))
                indexes_used[k] = True

                y_tilde = X[k] @ beta

                # TODO: Implement classes in linear_models.py in a way where this can come from it
                cost_diff = np.dot(X[k].T, 2/(data_size/mini_batches)*(y_tilde - data['y_train'][k][:,np.newaxis])) + 2*_lambda*beta

                v = momentum*v + learning_rate_func(step)*cost_diff
                beta = beta - v

                step += 1
            errors.loc[learning_rate_name][epoch] = metrics.MSE(X @ beta, data['y_train'][:,np.newaxis])


    return errors

if __name__ == '__main__':
    import real_terrain

    data = real_terrain.get_data(20)

    epochs = 50
    X_test = tune.poly_design_matrix(6, data['x_train'])

    learning_rate_iter = [(lambda t: learning_rate, f"flat_{learning_rate}") for learning_rate in np.logspace(-3, -1, 3)]
    errors = sgd(data, 20, epochs, learning_rate_iter, 6, momentum=0.5, _lambda=0.01, initial_conditions=50)

    for j, (_, learning_rate_name) in enumerate(learning_rate_iter):
        error_for_learning_rate = errors.loc[learning_rate_name]
        min_errors = error_for_learning_rate.min(axis=0)
        max_errors = error_for_learning_rate.max(axis=0)
        best_init_condition = error_for_learning_rate[epochs].idxmin()
        best_errors = error_for_learning_rate.loc[best_init_condition]
        plotting.plot_MSE_and_CI(best_errors, max_errors, min_errors, color_MSE='C'+str(j+1), color_shading='C'+str(j+1))

    model = linear_models.LinearRegression()
    model.fit(X_test, data['y_train'])
    plt.plot([1, epochs], np.repeat(metrics.MSE(model.predict(X_test), data['y_train']), 2), label="Analytical")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.yscale("log")

    modelnames = np.array([f"Learning rate {learning_rate_name}" for _, learning_rate_name in learning_rate_iter])
    modelnumbers = np.array([str(i) for i in range(1, len(learning_rate_iter) + 1)])
    colors = np.array(["C"] * len(learning_rate_iter), dtype=object) + modelnumbers
    handler_map = {}
    for i in range(len(colors)):
        handler_map[i] = plotting.LegendObject(colors[i])
    plt.legend(list(range(len(learning_rate_iter))), modelnames, handler_map=handler_map)

    #plt.savefig("../plots/sgd_different_learning_rates")
    plt.show()

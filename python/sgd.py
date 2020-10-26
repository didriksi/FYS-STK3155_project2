import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import tune
import metrics
import linear_models
import plotting

def sgd(data, learning_rate_iter, polynomials, epochs=50, mini_batch_size=1, momentum=0, _lambda=0, initial_conditions=1, metric=metrics.MSE):
    """Stochastic gradient descent

    Parameters:
    -----------
    data:       dict
                All training and testing data, as constructed by data_handling.make_data_dict.
    learning_rate_iter:
                iterable ((int -> float), str)
                Iterable object of a tuple of functions that takes in a timestep, and returns a learning rate,
                and a str indication name of learning rate function.
    polynomials:int
                Maximum degree of polynomials in design matrix
    epochs:     int
                Number of epochs. 50 by default.
    mini_batch_size:
                int
                Size of mini-batches. 1 by default.
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


    data_size = data['x_train'].shape[0]
    assert data_size%mini_batch_size == 0, "Data must be divisible with mini_batch_size"

    step = 0
    X = tune.poly_design_matrix(polynomials, data['x_train'])

    beta = np.random.randn(X.shape[1], initial_conditions)
    # TODO: Vectorise this loop away. --Seems difficult as long as it's a function that might include other factors than step
    for j, [learning_rate_func, learning_rate_name] in enumerate(learning_rate_iter):
        for epoch in range(1, epochs + 1):
            print("\r        ", j+1, '/ 3 |  epoch: 0| ' + "-" * (epoch // 5) + " "*(20 - epoch // 5), f"|{epoch:.0f}", end="")
            v = np.zeros_like(beta)
            indexes_used = np.zeros(data_size, dtype=bool)
            for mini_batch in range(1, int(data_size/mini_batch_size) + 1):
                k = np.random.choice(np.arange(data_size)[indexes_used == False], size=(mini_batch_size))
                indexes_used[k] = True

                y_tilde = X[k] @ beta

                # TODO: Implement classes in linear_models.py in a way where this can come from it
                cost_diff = np.dot(X[k].T, 2/mini_batch_size*(y_tilde - data['y_train'][k][:,np.newaxis])) + 2*_lambda*beta

                v = momentum*v + learning_rate_func(step)*cost_diff
                beta = beta - v

                step += 1
            errors.loc[learning_rate_name][epoch] = metrics.MSE(X @ beta, data['y_train'][:,np.newaxis])
        print("")

    return errors

def plot_sgd(data, title, learning_rate_iter, polynomials, subplot_parameters, **sgd_kwargs):
    """Plots Stochastic Gradient Descent error with confidence intervals, for different parameters.

    Parameters:
    -----------
    data:       dict
                All training and testing data, as constructed by data_handling.make_data_dict.
    title:      str or (str, str)
                Title for entire plot and filename. If list of two strings the second element is filename.
    learning_rate_iter:
                iterable ((int -> float), str)
                Iterable object of a tuple of functions that takes in a timestep, and returns a learning rate,
                and a str indication name of learning rate function.
    polynomials:int
                Maximum degree of polynomials in design matrix
    subplot_parameters:
                list of dicts
                List with each element being a dictionary with keyword arguments sent to sgd to make a dataset.
    **sgd_kwargs:
                kwargs
                Additional kwargs that are shared between the different subplots, sent to sgd.
    """

    X_train = tune.poly_design_matrix(polynomials, data['x_train'])
    kwargs = {'momentum': 0.5, '_lambda': 0.01, 'initial_conditions': 50, 'epochs': 100}
    kwargs.update(sgd_kwargs)

    plots = []
    for subplot_parameter in subplot_parameters:
        print(f"Mini batch size {subplot_parameter['mini_batch_size']}")
        print(" Learning rate |  Progress")

        kwargs.update(subplot_parameter)
        
        errors = sgd(data, learning_rate_iter, polynomials, **kwargs)
        y = []
        for j, (_, learning_rate_name) in enumerate(learning_rate_iter):
            error_for_learning_rate = errors.loc[learning_rate_name]
            min_errors = error_for_learning_rate.min(axis=0).to_numpy()
            max_errors = error_for_learning_rate.max(axis=0).to_numpy()
            best_index = error_for_learning_rate[kwargs['epochs']].idxmin()
            best_errors = error_for_learning_rate.loc[best_index].to_numpy()
            y.append((np.concatenate((min_errors[np.newaxis,:], best_errors[np.newaxis,:], max_errors[np.newaxis,:]), axis=0), {'color_MSE': 'C'+str(j+1), 'color_shading': 'C'+str(j+1)}))

        model = linear_models.LinearRegression()
        model.fit(X_train, data['y_train'])
        y.append((np.repeat(metrics.MSE(model.predict(X_train), data['y_train']), kwargs['epochs']), plotting.simple_plotter, {'label': "Analytical"}))

        plots.append([f"$\\lambda$: {subplot_parameter['_lambda']}. Mini batch size: {subplot_parameter['mini_batch_size']}", np.arange(1,kwargs['epochs']+1), y])

    learning_rate_enums = np.arange(len(learning_rate_iter))
    modelnames = [f"Learning rate {learning_rate_name}" for _, learning_rate_name in learning_rate_iter]
    colors = [f"C{number}" for number in learning_rate_enums + 1]
    handler_map = {i: plotting.LegendObject(colors[i]) for i in range(len(colors))}

    side_by_side_parameters  = {
        'plotter': plotting.confidence_interval_plotter,
        'axis_labels': ('Epoch', 'MSE'),
        'title': title,
        'yscale': 'log',
        'legend': [[learning_rate_enums, modelnames], {'handler_map': handler_map}]
    }
    plotting.side_by_side(*plots, **side_by_side_parameters)

if __name__ == '__main__':
    import real_terrain
    data = real_terrain.get_data(20)
    polynomials = 8

    learning_rate_iter = [(lambda t: learning_rate, f"flat_{learning_rate}") for learning_rate in np.logspace(-3, -1, 3)]

    title = ['Confidence interval for different learning rates and mini-batch sizes', 'conf_interval']

    subplot_parameters = [{'_lambda': _lambda, 'mini_batch_size': mini_batch_size} for _lambda in [0, 0.01] for mini_batch_size in [10, 20, 40]]
    plot_sgd(data, title, learning_rate_iter, polynomials, subplot_parameters, epochs=40)

    

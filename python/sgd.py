import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import tune
import metrics
import linear_models
import plotting

def sgd(model_x, y, learning_rates, model=linear_models.LinearRegression(), epochs=50, mini_batch_size=1, init_conds=1, metric=metrics.MSE, **model_kwargs):
    """Stochastic gradient descent

    Parameters:
    -----------
    model_x:    array of shape (n, k)
                Array with data usable by model, with each datapoint being a row.
    y:          array of shape (n, )
                Array with target data.
    learning_rates:
                iterable ((int -> float), str)
                Iterable object of a tuple of functions that takes in a timestep, and returns a learning rate,
                and a str indication name of learning rate function.
    model:      object
                Instance of class with a compile, set_learning_rate, predict and a update_parameters method.
    epochs:     int
                Number of epochs. 50 by default.
    mini_batch_size:
                int
                Size of mini-batches. 1 by default.
    init_conds:
                int
                Amount of attempts with different random initial conditions.
    metric:     array of shape(n, ), array of shape(n, ) -> float
                Should take in y and y_tilde, and return the total error.
    **model_kwargs:
                Keywoard arguments passed to model.compile
    
    Returns:
    --------
    errors:     pd.DataFrame
                Dataframe with initial conditions and learning rates as multiindex, and epochs as column
    """

    data_size = model_x.shape[0]
    indexes = np.arange(data_size)

    errors_index = pd.MultiIndex.from_product([
        [learning_rate_name for _, learning_rate_name in learning_rates],
        range(init_conds)],
        names = ['Learning rate number', 'Initial condition number']
        )
    errors = pd.DataFrame(dtype=float, index=errors_index, columns=range(1, epochs + 1))
    errors.sort_index(inplace=True)

    model.compile((model_x.shape[1], init_conds), **model_kwargs)

    for j, [learning_rate, learning_rate_name] in enumerate(learning_rates):
        model.set_learning_rate(learning_rate)
        for epoch in range(1, epochs + 1):
            print("\r        ", j+1, '/ 3 |  epoch: 0| ' + "-" * (epoch // 5) + " "*(20 - epoch // 5), f"|{epoch:.0f}", end="")
            errors.loc[learning_rate_name][epoch] = metrics.MSE(model.predict(model_x), y[:,np.newaxis]) # Check shapes
            
            np.random.shuffle(indexes)
            mini_batches = indexes.reshape(-1, mini_batch_size)
            for mini_batch in mini_batches:
                y_tilde = model.predict(model_x[mini_batch])
                gradient = model.gradient(model_x[mini_batch], y[mini_batch], y_tilde)
                model.update_parameters(gradient)

        print("")

    return errors

def plot_sgd(model_x, y, title, learning_rates, subplots_kwargs, **sgd_kwargs):
    """Plots Stochastic Gradient Descent error with confidence intervals, for different parameters.

    Parameters:
    -----------
    model_x:    array of shape (n, k)
                Array with data useable by model, with each datapoint being a row.
    y:          array of shape (n, )
                Array with target data.
    title:      str or (str, str)
                Title for entire plot and filename. If list of two strings the second element is filename.
    learning_rates:
                iterable ((int -> float), str)
                Iterable object of a tuple of functions that takes in a timestep, and returns a learning rate,
                and a str indication name of learning rate function.
    subplots_kwargs:
                list of dicts
                List with each element being a dictionary with keyword arguments sent to sgd.
    **sgd_kwargs:
                kwargs
                Additional kwargs that are shared between the different subplots, sent to sgd.
    """

    kwargs = {'momentum': 0.5, 'init_conds': 50, 'epochs': 100}
    kwargs.update(sgd_kwargs)

    plots = []
    for subplot_kwargs in subplots_kwargs:
        print(f"Mini batch size {subplot_kwargs['mini_batch_size']}")
        print(" Learning rate |  Progress")

        kwargs.update(subplot_kwargs)

        errors = sgd(model_x, y, learning_rates, **kwargs)

        plot_y = []
        for j, (_, learning_rate_name) in enumerate(learning_rate_iter):
            error_for_learning_rate = errors.loc[learning_rate_name]
            min_errors = error_for_learning_rate.min(axis=0).to_numpy()
            max_errors = error_for_learning_rate.max(axis=0).to_numpy()
            best_index = error_for_learning_rate[kwargs['epochs']].idxmin()
            best_errors = error_for_learning_rate.loc[best_index].to_numpy()

            plot_y.append((np.concatenate((min_errors[np.newaxis,:], best_errors[np.newaxis,:], max_errors[np.newaxis,:]), axis=0), {'color_MSE': 'C'+str(j+1), 'color_shading': 'C'+str(j+1)}))

        model = linear_models.LinearRegression()
        model.fit(model_x, y)
        plot_y.append((np.repeat(metrics.MSE(model.predict(model_x), y), kwargs['epochs']), plotting.simple_plotter, {'label': "Analytical"}))

        plots.append([f"Model: {subplot_kwargs['model'].name}. Mini batch size: {subplot_kwargs['mini_batch_size']}", np.arange(1,kwargs['epochs']+1), plot_y])

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
    X_train = tune.poly_design_matrix(polynomials, data['x_train'])
    print(X_train.shape)

    learning_rate_iter = [(lambda t: learning_rate, f"flat_{learning_rate}") for learning_rate in np.logspace(-3, -1, 3)]

    title = ['Confidence interval for different learning rates and mini-batch sizes', 'conf_interval']

    models = [linear_models.LinearRegression(), linear_models.RegularisedLinearRegression("Ridge", linear_models.beta_ridge)]
    models[1].set_lambda(0.01)
    subplots_kwargs = [{'model': model, 'mini_batch_size': mini_batch_size} for model in models for mini_batch_size in [10, 20, 40]]
    
    plot_sgd(X_train, data['y_train'], title, learning_rate_iter, subplots_kwargs, epochs=40)
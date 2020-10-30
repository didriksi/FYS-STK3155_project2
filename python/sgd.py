import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys

import tune
import metrics
import linear_models
import plotting

def sgd(model, x, y, epochs=50, mini_batch_size=1, metric=metrics.MSE):
    """Stochastic gradient descent

    Parameters:
    -----------
    model:      object
                Instance of class with a predict(x) and an update_parameters(x, y, y_tilde) method.
    x:          array of shape (n, k)
                Array with data usable by model, with each datapoint being a row.
    y:          array of shape (n, )
                Array with target data.
    epochs:     int
                Number of epochs. 50 by default.
    mini_batch_size:
                int
                Size of mini-batches. 1 by default.
    metric:     array of shape(n, ), array of shape(n, ) -> float
                Should take in y and y_tilde, and return the total error.
    
    Returns:
    --------
    model:      object
                Trained version of model sent to function.
    errors:     array
                Array with initial conditions and epochs
    """

    data_size = x.shape[0]
    indexes = np.arange(data_size)
    errors = np.empty((model.parallell_runs, epochs)) # TODO: Change order?

    for epoch in range(epochs):
        print("\r        ", '/ 3 |  epoch: 0| ' + "-" * (epoch+1 // 5) + " "*(20 - epoch+1 // 5), f"|{epoch+1:.0f}", end="")
        errors[:,epoch] = metrics.MSE(model.predict(x), y[:,np.newaxis])
        
        np.random.shuffle(indexes)
        mini_batches = indexes.reshape(-1, mini_batch_size)
        for mini_batch in mini_batches:
            y_tilde = model.predict(x[mini_batch])
            model.update_parameters(x[mini_batch], y[mini_batch], y_tilde)

    print("")

    return model, errors

def sgd_on_models(x, y, title, *subplots, **sgd_kwargs):
    """Does Stochastic Gradient Descent on models with different parameters, and returns errors DataFrame.

    The heart of the pipeline make_models->sgd_on_models->plot_sgd_errors, where sgd and plotting.side_by_side
    is also called. The pipeline is very flexible, and allows plotting of all kinds of combinations of models
    and parameters with only ~15 lines of code, but is maybe a bit too big and general. It would definitely
    have saved time on this project to just make specific code for the plots we wanted, instead of making this
    machine, and it would perhaps have been easier to read as well.

    Parameters:
    -----------
    x:          array of shape (n, k)
                Array with data useable by model, with each datapoint being a row.
    y:          array of shape (n, )
                Array with target data.
    title:      str or (str, str)
                Title for entire plot and filename. If list of two strings the second element is filename.
    *subplots:  (models, sgd kwargs)
                Each tuples first element is a list of models to be tested together, while the second is kwargs passed to sgd().
                The results is plotted as a subplot.
    **sgd_kwargs:
                kwargs
                Additional kwargs that are shared between the different subplots, sent to sgd.

    Returns:
    --------
    errors_df:  pd.DataFrame
                Dataframe with titles, initial conditions and labels as multiindex, and epochs as column
    """

    default_sgd_kwargs = {'epochs': 100}
    default_sgd_kwargs.update(sgd_kwargs)

    errors = np.empty((len(subplots), len(subplots[0][0]), subplots[0][0][0].parallell_runs, default_sgd_kwargs['epochs']))

    labels_list = []
    title_dicts = []
    for i, (models, subplot_sgd_kwargs) in enumerate(subplots):
        sgd_kwargs = default_sgd_kwargs.copy()
        sgd_kwargs.update(subplot_sgd_kwargs)

        subplot_title_dict = {'key': i}
        subplot_title_dict.update(subplot_sgd_kwargs)
        subplot_labels = []
        for j, model in enumerate(models):
            subplot_dict = model.property_dict
            subplot_dict.update(sgd_kwargs)
            subplot_labels.append(subplot_dict)

            model, errors[i,j] = sgd(model, x, y, **sgd_kwargs)

        # Make dictionaries and eventually strings describing the unique aspects of each subplot and subsubplot
        filtered_labels_dicts, title_dict = filter_dicts(subplot_labels)
        labels_list.append([textify_dict(filtered_labels_dict) for filtered_labels_dict in filtered_labels_dicts])
        title_dicts.append(title_dict)

    subplot_titles_dicts, plot_title_dict = filter_dicts(title_dicts)
    subplot_titles = [textify_dict(subplot_titles_dict) for subplot_titles_dict in subplot_titles_dicts]
    plot_title = textify_dict(plot_title_dict)

    errors_index_tuple = [(title, label, parallell_run) for title, labels in zip(subplot_titles, labels_list) for label in labels for parallell_run in range(model.parallell_runs)]
    errors_index = pd.MultiIndex.from_tuples(errors_index_tuple, names=['Titles', 'Labels', 'Parallell runs'])
    errors_df = pd.DataFrame(data=errors.reshape(-1, default_sgd_kwargs['epochs']), index=errors_index, columns=range(1, default_sgd_kwargs['epochs'] + 1))
    errors_df.sort_index(inplace=True)

    errors_df.to_csv('../dataframes/errors.csv')

    return errors_df

def filter_dicts(dicts):
    """Splits list of dictionaries into what is common, and what is unique with them.

    Takes in a list of dictionaries, and goes through them all, removing keys where all of the dictionaries have
    that key, and the associated values are all the same. The removed values are stored in a dict, and returnes as well.

    Parameters:
    -----------
    dicts:      list of dictionaries
                Dictionaries you want filtered.

    Returns:
    --------
    filtered_dicts:
                list of dictionaries
                Dictionaries with only the unique key/value combinations in them
    common:     dictionarie
                The key/value combinations that were shared between all the input dictionaries
    """

    common = {}
    unique_keys = {}
    
    keys = [[i, key, value] for i, label_dict in enumerate(dicts) for key, value in label_dict.items()]
    for i, key, value in keys:
        if key not in unique_keys:
            unique_keys[key] = [[i, value]]
        else:
            unique_keys[key].append([i, value])

    for unique_key, value_list in unique_keys.items():
        if len(value_list) == len(dicts):
            equals = 0
            for e, _ in enumerate(value_list[:-1]):
                if value_list[e][1] == value_list[e+1][1]:
                    equals += 1
            if equals == len(value_list)-1:
                common[unique_key] = value_list[0][1]
                for dict_to_filter in dicts:
                    del dict_to_filter[unique_key]
    return dicts, common

def textify_dict(dictionary):
    """Takes in a dictionary, and makes it into a relatively tidy string.

    Has special rules for some cases, where it for example omits the key name
    because the value is explanatory enough on itself.

    Parameters:
    -----------
    dictionary: dictionarie
                Dictionary you want turned into a string

    Returns:
    --------
    string:     string
                String which describes input dict
    """

    string_list = []
    for key, value in dictionary.items():
        if key == 'model_name':
            string_list.append(value)
        elif key == '_lambda':
            string_list.append(f"($\\lambda ${value})")
        else:
            key = key.replace('_', ' ').capitalize()
            string_list.append(f"{key}: {value}")
    return ' '.join(string_list)

def plot_sgd_errors(sgd_errors_df, title):
    """Takes a dataframe of errors, and formats it in a way that plotting.side_by_side can handle.

    Parameters:
    -----------
    errors_df:  pd.DataFrame
                Dataframe with titles, initial conditions and labels as multiindex, and epochs as column
    """
    plots = []
    for subplot_name, subplot_errors in sgd_errors_df.groupby(level=0):
        y = []
        labels = []
        for j, (label_names, label_errors) in enumerate(subplot_errors.groupby(level=1)):
            labels.append(label_names)
            min_errors = label_errors.min(axis=0).to_numpy()[np.newaxis,:]
            max_errors = label_errors.max(axis=0).to_numpy()[np.newaxis,:]
            best_index = label_errors.iloc[slice(None), -1].idxmin()
            best_errors = label_errors.loc(axis=0)[slice(None), slice(None), best_index].to_numpy()

            y_ = np.concatenate((min_errors, best_errors, max_errors), axis=0)
            y.append((y_, {'color_MSE': 'C'+str(j+1), 'color_shading': 'C'+str(j+1)}))

            learning_rate_enums = np.arange(len(labels))
            colors = [f"C{number}" for number in learning_rate_enums + 1]
            handler_map = {i: plotting.LegendObject(colors[i]) for i in range(len(colors))}
            legend = [learning_rate_enums, labels], {'handler_map': handler_map}
        
        plots.append([subplot_name, subplot_errors.columns, y, legend])

    side_by_side_parameters  = {
        'plotter': plotting.confidence_interval_plotter,
        'axis_labels': ('Epoch', 'MSE'),
        'title': title,
        'yscale': 'log',
    }
    plotting.side_by_side(*plots, **side_by_side_parameters)

def make_models(model_class, common_kwargs, subplot_uniques, subplot_copies, subsubplot_uniques):
    """Make list of models with different parameters. Useful for passing to sgd_on_models.

    Parameters:
    -----------
    model_class:class
                Class (not instance of class) to be called with parameters later specified
    common_kwargs:
                dictionary
                Keyword arguments shared by all the models
    subplot_uniques:
                list of dictionaries
                Each dictionary represents keyword arguments shared by the models in a subplot
    subplot_copies:
                int
                Amount of copies of each model. Used if you want to test the same models on different environments
    subsubplot_uniques:
                list of dictionaries
                Keyword arguments used to differentiate models within subplot

    Returns:
    --------
    models:     list of instances of model:class
                Nested list with subplot as first level, and subsubplots at second level
    """
    models = []
    for _ in range(subplot_copies):
        for subplot_unique_kwargs in subplot_uniques:
            subplot_kwargs = common_kwargs.copy()
            subplot_kwargs.update(subplot_unique_kwargs)
            subplot_models = []
            for subsubplot_unique_kwargs in subsubplot_uniques:
                subsubplot_kwargs = subplot_kwargs.copy()
                subsubplot_kwargs.update(subsubplot_unique_kwargs)
                subplot_models.append(model_class(**subsubplot_kwargs))
            models.append(subplot_models)
    return models



if __name__ == '__main__':
    def conf_interval_plot(data):
        polynomials = 8
        X_train = tune.poly_design_matrix(polynomials, data['x_train'])
        
        title = ['Confidence interval for different learning rates and mini-batch sizes', 'conf_interval']
        
        common_ols_kwargs = {'momentum': 0.5, 'init_conds': 50, 'x_shape': X_train.shape[1]}
        common_ridge_kwargs = {'name': 'Ridge', 'beta_func': linear_models.beta_ridge, 'momentum': 0.5, 'init_conds': 50, 'x_shape': X_train.shape[1]}
        subplot_ols_uniques = [{}]
        subplot_ridge_uniques = [{'_lambda': 0.01}, {'_lambda': 0.001}]
        subsubplot_linear_uniques = [{'learning_rate': learning_rate} for learning_rate in np.logspace(-3, -1, 3)]

        unique_sgd_kwargs = [{'mini_batch_size': mini_batch_size} for mini_batch_size in [10, 20, 40]]

        ols_models = make_models(
            linear_models.LinearRegression,
            common_ols_kwargs,
            subplot_ols_uniques,
            len(unique_sgd_kwargs),
            subsubplot_linear_uniques
            )
        ridge_models = make_models(
            linear_models.RegularisedLinearRegression,
            common_ridge_kwargs,
            subplot_ridge_uniques,
            len(unique_sgd_kwargs),
            subsubplot_linear_uniques
            )

        subplots = [(models, sgd_kwargs) for models, sgd_kwargs in zip(ols_models + ridge_models, unique_sgd_kwargs*3)]

        errors = sgd_on_models(X_train, data['y_train'], title, *subplots, epochs=150)

        plot_sgd_errors(errors)

    def momemtun_plot(data):
        polynomials = 8
        X_train = tune.poly_design_matrix(polynomials, data['x_train'])

        title = ['Confidence interval for different momentums and learning rates', 'momentum']
        
        common_ridge_kwargs = {'name': 'Ridge', 'beta_func': linear_models.beta_ridge, 'momentum': 0.5, 'init_conds': 50, 'x_shape': X_train.shape[1]}
        subplot_ridge_uniques = [{'momentum': 0.9}, {'momentum': 0.6}, {'momentum': 0.3}, {'momentum': 0.}]
        subsubplot_linear_uniques = [{'learning_rate': learning_rate} for learning_rate in np.logspace(-3, -1, 3)]

        unique_sgd_kwargs = [{'mini_batch_size': 10}]

        ridge_models = make_models(
            linear_models.RegularisedLinearRegression,
            common_ridge_kwargs,
            subplot_ridge_uniques,
            len(unique_sgd_kwargs),
            subsubplot_linear_uniques
            )

        subplots = [(models, sgd_kwargs) for models, sgd_kwargs in zip(ridge_models, unique_sgd_kwargs*len(ridge_models))]

        errors = sgd_on_models(X_train, data['y_train'], title, *subplots, epochs=150)

        plot_sgd_errors(errors, title)

    def neural_plot(data):
        pass

    import real_terrain
    data = real_terrain.get_data(20)

    if 'conf' in sys.argv:
        conf_interval_plot(data)
    if 'momentum' in sys.argv:
        momemtun_plot(data)
    if 'neural' in sys.argv:
        neural_plot(data)







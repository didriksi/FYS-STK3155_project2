import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys

import tune
import metrics
import linear_models
import plotting
import neural_model
import helpers

def sgd(model, x_train, x_test, y_train, y_test, epochs=200, epochs_without_progress=50, mini_batch_size=1, metric=metrics.MSE):
    """Stochastic gradient descent

    Parameters:
    -----------
    model:      object
                Instance of class with a predict(x) and an update_parameters(x, y, y_tilde) method.
    x_train:    array of shape (n, k)
                Array with data usable by model, with each datapoint being a row. Data used to train the model.
    x_test:     array of shape (n, k)
                Array with data usable by model, with each datapoint being a row. Used to test the model, and calculate error.
    y_train:    array of shape (n, )
                Array with target data used to train the model.
    y_test:     array of shape (n, )
                Array with target data used to test the model, and calculate error.
    epochs:     int
                Number of epochs. 200 by default.
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
    data_size = x_train.shape[0]
    indexes = np.arange(data_size)
    errors = np.zeros((model.parallell_runs, epochs))

    last_index = data_size//mini_batch_size * mini_batch_size

    for epoch in range(epochs):
        errors[:,epoch] = metric(y_test, model.predict(x_test))
        if epoch % epochs_without_progress == 0 and epoch >= epochs_without_progress:
            if epoch == epochs_without_progress:
                prev_error = np.mean(errors[:,epoch-epochs_without_progress:epoch])
            else:
                current_error = np.mean(errors[:,epoch-epochs_without_progress:epoch])
                if current_error > prev_error:
                    errors[:,epoch:] = errors[:,[epoch]]
                    print(" -> Breaking at epoch", epoch)
                    break
                prev_error = current_error

        np.random.shuffle(indexes)
        mini_batches = indexes[:last_index].reshape(-1, mini_batch_size)
        for mini_batch in mini_batches:
            y_tilde = model.predict(x_train[mini_batch])
            model.update_parameters(x_train[mini_batch], y_train[mini_batch], y_tilde)

    return model, errors

def sgd_on_models(x_train, x_test, y_train, y_test, *subplots, **sgd_kwargs):
    """Does Stochastic Gradient Descent on models with different parameters, and returns errors DataFrame.

    The heart of the pipeline make_models->sgd_on_models->plot_sgd_errors, where sgd and plotting.side_by_side
    is also called. The pipeline is very flexible, and allows plotting of all kinds of combinations of models
    and parameters with only ~15 lines of code, but is maybe a bit too big and general. It would definitely
    have saved time on this project to just make specific code for the plots we wanted, instead of making this
    machine, and it would perhaps have been easier to read as well.

    Parameters:
    -----------
    x_train:    array of shape (n, k)
                Array with data usable by model, with each datapoint being a row. Data used to train the model.
    x_test:     array of shape (n, k)
                Array with data usable by model, with each datapoint being a row. Used to test the model, and calculate error.
    y_train:    array of shape (n, )
                Array with target data used to train the model.
    y_test:     array of shape (n, )
                Array with target data used to test the model, and calculate error.
    *subplots:  (models, sgd kwargs)
                Each tuples first element is a list of models to be tested together, while the second is kwargs passed to sgd().
                The results is plotted as a subplot. If models is a list of lists, the first element is the model, and the rest is
                x_train, x_test, y_train, y_test for that specific model.
    **sgd_kwargs:
                kwargs
                Additional kwargs that are shared between the different subplots, sent to sgd.

    Returns:
    --------
    errors_df:  pd.DataFrame
                Dataframe with titles, initial conditions and labels as multiindex, and epochs as column
    plot_title: str
                String with whatever is common between all the plots
    subplots:   list of (models, sgd kwargs)
                The same as input subplots, just with all models trained.
    """
    
    default_sgd_kwargs = {'epochs': 100}
    default_sgd_kwargs.update(sgd_kwargs)

    errors = np.empty((len(subplots), len(subplots[0][0]), subplots[0][0][0].parallell_runs, default_sgd_kwargs['epochs']))

    step = 0
    end_step = len([model for (models, _) in subplots for model in models])

    y_test = y_test if y_test.ndim == 2 else y_test[:,np.newaxis]
    y_train = y_train if y_train.ndim == 2 else y_train[:,np.newaxis]

    labels_list = []
    title_dicts = []
    for i, (models, subplot_sgd_kwargs) in enumerate(subplots):
        sgd_kwargs = default_sgd_kwargs.copy()
        sgd_kwargs.update(subplot_sgd_kwargs)

        subplot_title_dict = {'key': i}
        subplot_title_dict.update(subplot_sgd_kwargs)
        subplot_labels = []
        for j, model in enumerate(models):
            if isinstance(model, list):
                data = model[1:]
                model = model[0]
            else:
                data = [x_train, x_test, y_train, y_test]

            subplot_dict = model.property_dict
            subplot_dict.update(sgd_kwargs)
            subplot_labels.append(subplot_dict)

            step += 1
            print(f"\r |{'='*(step*50//end_step)}{' '*(50-step*50//end_step)}| {step/end_step:.2%}", end="", flush=True)

            model, errors[i,j] = sgd(model, *data, **sgd_kwargs)

        # Make dictionaries and eventually strings describing the unique aspects of each subplot and subsubplot
        filtered_labels_dicts, title_dict = helpers.filter_dicts(subplot_labels)
        labels_list.append([helpers.textify_dict(filtered_labels_dict) for filtered_labels_dict in filtered_labels_dicts])
        title_dicts.append(title_dict)

    subplot_titles_dicts, plot_title_dict = helpers.filter_dicts(title_dicts)
    subplot_titles = [helpers.textify_dict(subplot_titles_dict) for subplot_titles_dict in subplot_titles_dicts]
    plot_title = helpers.textify_dict(plot_title_dict)

    errors_index_tuple = [(title, label, parallell_run) for title, labels in zip(subplot_titles, labels_list) for label in labels for parallell_run in range(model.parallell_runs)]
    errors_index = pd.MultiIndex.from_tuples(errors_index_tuple, names=['Titles', 'Labels', 'Parallell runs'])
    errors_df = pd.DataFrame(data=errors.reshape(-1, default_sgd_kwargs['epochs']), index=errors_index, columns=range(1, default_sgd_kwargs['epochs'] + 1))
    errors_df.sort_index(inplace=True)

    errors_df.to_csv('../dataframes/errors.csv')

    print("")

    metric_string = metrics.MSE.__doc__ if 'metric' not in sgd_kwargs else sgd_kwargs['metric'].__doc__

    return errors_df, plot_title, subplots, metric_string

def plot_sgd_errors(sgd_errors_df, title, metric_string='MSE'):
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
            y.append((y_, {'color': 'C'+str(j+1)}))

            label_enum = np.arange(len(labels))
            colors = [f"C{number}" for number in label_enum + 1]
            handler_map = {i: plotting.LegendObject(colors[i]) for i in range(len(colors))}
            legend = [label_enum, labels], {'handler_map': handler_map}
        
        plots.append([subplot_name, subplot_errors.columns, y, legend])

    side_by_side_parameters  = {
        'plotter': plotting.confidence_interval_plotter,
        'axis_labels': ('Epoch', metric_string),
        'title': title,
        'yscale': 'log',
    }
    plotting.side_by_side(*plots, **side_by_side_parameters)
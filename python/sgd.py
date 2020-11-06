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
    data_size = x_train.shape[0]
    indexes = np.arange(data_size)
    errors = np.zeros((model.parallell_runs, epochs))

    for epoch in range(epochs):
        errors[:,epoch] = metric(model.predict(x_test), y_test)
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
        mini_batches = indexes.reshape(-1, mini_batch_size)
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
                The results is plotted as a subplot.
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
            subplot_dict = model.property_dict
            subplot_dict.update(sgd_kwargs)
            subplot_labels.append(subplot_dict)

            step += 1
            print(f"\r |{'='*(step*50//end_step)}{' '*(50-step*50//end_step)}| {step/end_step:.2%}", end="", flush=True)

            model, errors[i,j] = sgd(model, x_train, x_test, y_train, y_test, **sgd_kwargs)

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

    print("")
    return errors_df, plot_title, subplots

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
            y.append((y_, {'color': 'C'+str(j+1)}))

            label_enum = np.arange(len(labels))
            colors = [f"C{number}" for number in label_enum + 1]
            handler_map = {i: plotting.LegendObject(colors[i]) for i in range(len(colors))}
            legend = [label_enum, labels], {'handler_map': handler_map}
        
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
    def beta_initialiser(shape):
        """3*stdn"""
        return 3*np.random.randn(*shape)

    def learning_rate_adaptation(base_learning_rate, step_factor):
        def learning_rate(step):
            f"""{base_learning_rate}/(1+step*{step_factor})"""
            return base_learning_rate/(1+step*step_factor)
        return learning_rate

    def conf_interval_plot(data):
        polynomials = 8
        X_train = tune.poly_design_matrix(polynomials, data['x_train'])
        X_validate = tune.poly_design_matrix(polynomials, data['x_validate'])

        common_ols_kwargs = {'momentum': 0.5, 'init_conds': 50, 'x_shape': X_train.shape[1], 'beta_initialiser': beta_initialiser}
        common_ridge_kwargs = {'name': 'Ridge', 'beta_func': linear_models.beta_ridge, 'momentum': 0.5, 'init_conds': 50, 'x_shape': X_train.shape[1], 'beta_initialiser': beta_initialiser}
        subplot_ols_uniques = [{}]
        subplot_ridge_uniques = [{'_lambda': 0.01}, {'_lambda': 0.001}]
        subsubplot_linear_uniques = [{'learning_rate': learning_rate_adaptation(base_learning_rate, 1/400000)} for base_learning_rate in np.logspace(-3, -1, 3)]

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
        errors, subtitle, subplots = sgd_on_models(X_train, X_validate, data['y_train'], data['y_validate'], *subplots, epochs=300)

        title = ['Confidence interval for different learning rates and mini-batch sizes', 'conf_interval', subtitle]
        plot_sgd_errors(errors, title)

    def beta_variance(data):
        polynomials = 8
        X_train = tune.poly_design_matrix(polynomials, data['x_train'])
        X_validate = tune.poly_design_matrix(polynomials, data['x_validate'])
        
        common_ridge1_kwargs = {'_lambda': 0.01, 'name': 'Ridge', 'beta_func': linear_models.beta_ridge, 'momentum': 0.5, 'init_conds': 300, 'x_shape': X_train.shape[1], 'beta_initialiser': beta_initialiser}
        common_ridge2_kwargs = {'_lambda': 0.0001, 'name': 'Ridge', 'beta_func': linear_models.beta_ridge, 'momentum': 0.5, 'init_conds': 300, 'x_shape': X_train.shape[1], 'beta_initialiser': beta_initialiser}
        subplot_ridge_uniques = [{}]
        subsubplot_ridge_linear_uniques = [{'learning_rate': learning_rate_adaptation(5e-2, 1/100000)}]

        common_ols_kwargs = {'momentum': 0.5, 'init_conds': 300, 'x_shape': X_train.shape[1], 'beta_initialiser': beta_initialiser}
        subplot_ols_uniques = [{}]
        subsubplot_ols_linear_uniques = [{'learning_rate': learning_rate_adaptation(5e-2, 1/100000)}]

        unique_sgd_kwargs = [{'mini_batch_size': mini_batch_size} for mini_batch_size in [10, 20]]

        ridge_models1 = make_models(
            linear_models.RegularisedLinearRegression,
            common_ridge1_kwargs,
            [{}],
            len(unique_sgd_kwargs),
            subsubplot_ridge_linear_uniques
            )

        ridge_models2 = make_models(
            linear_models.RegularisedLinearRegression,
            common_ridge2_kwargs,
            [{}],
            len(unique_sgd_kwargs),
            subsubplot_ridge_linear_uniques
            )

        ols_models = make_models(
            linear_models.LinearRegression,
            common_ols_kwargs,
            [{}],
            len(unique_sgd_kwargs),
            subsubplot_ols_linear_uniques
            )

        models_list = ridge_models1 + ridge_models2 + ols_models 

        subplots = [(models, sgd_kwargs) for models, sgd_kwargs in zip(models_list, unique_sgd_kwargs*(len(models_list)))]
        errors, subtitle, subplots = sgd_on_models(X_train, X_validate, data['y_train'], data['y_validate'], *subplots, epochs=20000, epochs_without_progress=200)

        title = ['Convergence test', 'convergence', subtitle]
        plot_sgd_errors(errors, title)

        sgd_ridge1_betas = np.array([model.beta for models in ridge_models1 for model in models])
        sgd_ridge1_betas = sgd_ridge1_betas.transpose(1, 2, 0).reshape((sgd_ridge1_betas.shape[1], -1)).T

        sgd_ridge2_betas = np.array([model.beta for models in ridge_models2 for model in models])
        sgd_ridge2_betas = sgd_ridge2_betas.transpose(1, 2, 0).reshape((sgd_ridge2_betas.shape[1], -1)).T

        sgd_ols_betas = np.array([model.beta for models in ols_models for model in models])
        sgd_ols_betas = sgd_ols_betas.transpose(1, 2, 0).reshape((sgd_ols_betas.shape[1], -1)).T

        optimal_models = [linear_models.RegularisedLinearRegression("Ridge", linear_models.beta_ridge, _lambda=_lambda) for _lambda in (0.01, 0.)]
        optimal_betas = np.array([model.fit(X_train, data['y_train']) for model in optimal_models])[:,:,0]

        plots = [
            ['Ridge ($\\lambda 0.01$) trained by SGD', np.arange(1, optimal_betas.shape[1] + 1),
                [[sgd_ridge1_betas, {'notch': False, 'sym': ''}],
                [optimal_betas[0], plotting.scatter_plotter, {'label': 'Analytical $\\lambda$ 0.01'}],
                ]
            ],
            ['Ridge ($\\lambda 0.0001$) trained by SGD', np.arange(1, optimal_betas.shape[1] + 1),
                [[sgd_ridge2_betas, {'notch': False, 'sym': ''}],
                [optimal_betas[0], plotting.scatter_plotter, {'label': 'Analytical $\\lambda$ 0.01'}],
                ]
            ],
            ['OLS trained by SGD', np.arange(1, optimal_betas.shape[1] + 1),
                [[sgd_ols_betas, {'notch': False, 'sym': ''}],
                [optimal_betas[0], plotting.scatter_plotter, {'label': 'Analytical $\\lambda$ 0.01'}],
                ]
            ],
        ]

        side_by_side_parameters = {
            'plotter': plotting.box_plotter,
            'axis_labels': ('Beta parameter #', 'Values'),
            'title': ['$\\beta$ parameters', 'beta', subtitle],
        }
        plotting.side_by_side(*plots, **side_by_side_parameters)

    def momemtun_plot(data):
        polynomials = 8
        X_train = tune.poly_design_matrix(polynomials, data['x_train'])
        X_validate = tune.poly_design_matrix(polynomials, data['x_validate'])

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

        errors, subtitle, subplots = sgd_on_models(X_train, X_validate, data['y_train'], data['y_validate'], *subplots, epochs=300)

        title = ['Confidence interval for different momentums and learning rates', 'momentum', subtitle]

        plot_sgd_errors(errors, title)

    def neural_regression(data):
        common_kwargs = {'momentum': 0.5}
        subplot_uniques = [{'layers': [{'height': 2}, *hidden_layers, {'height': 1}]} for hidden_layers in [[{'height': hidden}] for hidden in [2,4]] + [[{'height': hidden}, {'height': hidden}] for hidden in [2, 3]]]
        subsubplot_uniques = [{'learning_rate': learning_rate_adaptation(base_learning_rate, 1/200000)} for base_learning_rate in np.logspace(-3, -1, 3)]

        unique_sgd_kwargs = [{'mini_batch_size': 20}]

        neural_models = make_models(
            neural_model.Network,
            common_kwargs,
            subplot_uniques,
            len(unique_sgd_kwargs),
            subsubplot_uniques
            )

        subplots = [(models, sgd_kwargs) for models, sgd_kwargs in zip(neural_models, unique_sgd_kwargs*len(neural_models))]

        errors, subtitle, subplots = sgd_on_models(data['x_train'], data['x_validate'], data['y_train'], data['y_validate'], *subplots, epochs=2000, epochs_without_progress=100)

        title = ['Neural model, regression on terrain data', 'neural_regression', subtitle]

        plot_sgd_errors(errors, title)

    def neural_classification(data):
        common_kwargs = {'momentum': 0.3}
        subplot_uniques = [{'layers': [{'height': 64}, {'height': 32}, {'height': 10}]}, {'name': 'Logistic', 'layers': [{'height': 64}, {'height': 10}]}] # , 'activation': neural_model.softmax, 'diff_activation': neural_model.softmax_diff
        #subplot_uniques = [{'layers': [{'height': 64}, {'height': hidden}, {'height': 10}]} for hidden in np.power(2, np.arange(4,7))] # , 'activation': neural_model.softmax, 'diff_activation': neural_model.softmax_diff
        #subsubplot_uniques = [{}]
        subsubplot_uniques = [{'learning_rate': learning_rate} for learning_rate in np.logspace(-3, -1, 3)]
        #subsubplot_uniques = [{'learning_rate': learning_rate, 'momentum': momentum} for learning_rate in np.logspace(-10, -6, 2) for momentum in [0., 0.4]]

        unique_sgd_kwargs = [{'mini_batch_size': 10}]

        neural_models = make_models(
            neural_model.Network,
            common_kwargs,
            subplot_uniques,
            len(unique_sgd_kwargs),
            subsubplot_uniques
            )

        subplots = [(models, sgd_kwargs) for models, sgd_kwargs in zip(neural_models, unique_sgd_kwargs*len(neural_models))]

        errors, subtitle, subplots = sgd_on_models(data['x_train'], data['x_validate'], data['y_train'], data['y_validate'], *subplots, epochs=600)

        title = ['Neural model, Classification test', 'neural_classification', subtitle]

        plot_sgd_errors(errors, title)

    import real_terrain
    import mnist

    terrainData = real_terrain.get_data(20)
    mnistData = mnist.get_data(0.6, 0.2)

    if 'conf' in sys.argv:
        conf_interval_plot(terrainData)
    if 'momentum' in sys.argv:
        momemtun_plot(terrainData)
    if 'neural' in sys.argv and 'reg' in sys.argv:
        neural_regression(terrainData)
    if 'neural' in sys.argv and 'class' in sys.argv:
        neural_classification(mnistData)
    if 'betas' in sys.argv:
        beta_variance(terrainData)

    le = learning_rate_adaptation(0.2, 0.001)
    def test():
        """test_doc"""
        return 0
    print(le.__doc__)
    print(test.__doc__)






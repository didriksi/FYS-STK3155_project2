import numpy as np
import sys
import tune

import learning_rate
import linear_models
import neural_model
import helpers
import sgd
import metrics
import activations

def conf_interval_plot(data, epochs=300):
    """Plots confidence intervals for different linear models on regression problem for real terrain"""
    polynomials = 8
    X_train = linear_models.poly_design_matrix(polynomials, data['x_train'])
    X_validate = linear_models.poly_design_matrix(polynomials, data['x_validate'])

    common_ols_kwargs = {'momentum': 0.5, 'init_conds': 50, 'x_shape': X_train.shape[1]}
    common_ridge_kwargs = {'name': 'Ridge', 'beta_func': linear_models.beta_ridge, 'momentum': 0.5, 'init_conds': 50, 'x_shape': X_train.shape[1]}
    subplot_ols_uniques = [{}]
    subplot_ridge_uniques = [{'_lambda': 0.01}, {'_lambda': 0.001}]
    subsubplot_linear_uniques = [{'learning_rate': learning_rate} for learning_rate in np.logspace(-3, -1, 3)]

    unique_sgd_kwargs = [{'mini_batch_size': mini_batch_size} for mini_batch_size in [10, 20, 40]]

    ols_models = helpers.make_models(
        linear_models.LinearRegression,
        common_ols_kwargs,
        subplot_ols_uniques,
        len(unique_sgd_kwargs),
        subsubplot_linear_uniques
        )
    ridge_models = helpers.make_models(
        linear_models.RegularisedLinearRegression,
        common_ridge_kwargs,
        subplot_ridge_uniques,
        len(unique_sgd_kwargs),
        subsubplot_linear_uniques
        )

    subplots = [(models, sgd_kwargs) for models, sgd_kwargs in zip(ols_models + ridge_models, unique_sgd_kwargs*3)]
    errors, subtitle, subplots, metrics_string = sgd.sgd_on_models(X_train, X_validate, data['y_train'], data['y_validate'], *subplots, epochs=epochs)

    title = ['Confidence interval for different learning rates and mini-batch sizes', 'conf_interval', subtitle]
    sgd.plot_sgd_errors(errors, title, metrics_string)

def beta_variance(data, epochs=20000, mini_batch_sizes=[10, 20]):
    """Plots variance of beta parameters for different linear models on regression problem for real terrain"""
    polynomials = 8
    X_train = linear_models.poly_design_matrix(polynomials, data['x_train'])
    X_validate = linear_models.poly_design_matrix(polynomials, data['x_validate'])

    total_steps = epochs * len(data['x_train'])//mini_batch_sizes[0]
    learning_rate_func = learning_rate.Learning_rate(base=5e-2, decay=1/100000).compile(total_steps)
    
    common_ridge1_kwargs = {'_lambda': 0.01, 'name': 'Ridge', 'beta_func': linear_models.beta_ridge, 'momentum': 0.5, 'init_conds': 300, 'x_shape': X_train.shape[1]}
    common_ridge2_kwargs = {'_lambda': 0.0001, 'name': 'Ridge', 'beta_func': linear_models.beta_ridge, 'momentum': 0.5, 'init_conds': 300, 'x_shape': X_train.shape[1]}
    subplot_ridge_uniques = [{}]
    subsubplot_ridge_linear_uniques = [{'learning_rate': learning_rate_func}]

    common_ols_kwargs = {'momentum': 0.5, 'init_conds': 300, 'x_shape': X_train.shape[1]}
    subplot_ols_uniques = [{}]
    subsubplot_ols_linear_uniques = [{'learning_rate': learning_rate_func}]

    unique_sgd_kwargs = [{'mini_batch_size': mini_batch_size} for mini_batch_size in mini_batch_sizes]

    ridge_models1 = helpers.make_models(
        linear_models.RegularisedLinearRegression,
        common_ridge1_kwargs,
        [{}],
        len(unique_sgd_kwargs),
        subsubplot_ridge_linear_uniques
        )

    ridge_models2 = helpers.make_models(
        linear_models.RegularisedLinearRegression,
        common_ridge2_kwargs,
        [{}],
        len(unique_sgd_kwargs),
        subsubplot_ridge_linear_uniques
        )

    ols_models = helpers.make_models(
        linear_models.LinearRegression,
        common_ols_kwargs,
        [{}],
        len(unique_sgd_kwargs),
        subsubplot_ols_linear_uniques
        )

    models_list = ridge_models1 + ridge_models2 + ols_models 

    subplots = [(models, sgd_kwargs) for models, sgd_kwargs in zip(models_list, unique_sgd_kwargs*(len(models_list)))]
    errors, subtitle, subplots, metrics_string = sgd.sgd_on_models(X_train, X_validate, data['y_train'], data['y_validate'], *subplots, epochs=epochs, epochs_without_progress=200)

    title = ['Convergence test', 'convergence', subtitle]
    sgd.plot_sgd_errors(errors, title, metrics_string)

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

def momentum_plot(data):
    """Plots effect of momentum for different linear models on regression problem for real terrain"""
    polynomials = 8
    X_train = linear_models.poly_design_matrix(polynomials, data['x_train'])
    X_validate = linear_models.poly_design_matrix(polynomials, data['x_validate'])

    common_ridge_kwargs = {'name': 'Ridge', 'beta_func': linear_models.beta_ridge, 'momentum': 0.5, 'init_conds': 50, 'x_shape': X_train.shape[1]}
    subplot_ridge_uniques = [{'momentum': 0.9}, {'momentum': 0.6}, {'momentum': 0.3}, {'momentum': 0.}]
    subsubplot_linear_uniques = [{'learning_rate': learning_rate} for learning_rate in np.logspace(-3, -1, 3)]

    unique_sgd_kwargs = [{'mini_batch_size': 10}]

    ridge_models = helpers.make_models(
        linear_models.RegularisedLinearRegression,
        common_ridge_kwargs,
        subplot_ridge_uniques,
        len(unique_sgd_kwargs),
        subsubplot_linear_uniques
        )

    subplots = [(models, sgd_kwargs) for models, sgd_kwargs in zip(ridge_models, unique_sgd_kwargs*len(ridge_models))]

    errors, subtitle, subplots, metrics_string = sgd.sgd_on_models(X_train, X_validate, data['y_train'], data['y_validate'], *subplots, epochs=300)

    title = ['Confidence interval for different momentums and learning rates', 'momentum', subtitle]

    sgd.plot_sgd_errors(errors, title, metrics_string)

def neural_regression(data, epochs=5000, epochs_without_progress=500, mini_batch_size=20):
    """Plots performance of different neural models on regression problem for real terrain"""
    polynomials = 8
    X_train = linear_models.poly_design_matrix(polynomials, data['x_train'])
    X_validate = linear_models.poly_design_matrix(polynomials, data['x_validate'])
    X_test = linear_models.poly_design_matrix(polynomials, data['x_test'])

    total_steps = epochs * len(data['x_train'])//mini_batch_size
    
    l_rate = learning_rate.Learning_rate(base=3e-2, decay=1/20000).ramp_up(10).compile(total_steps)

    sigmoid = {'name': 'Sigmoid',
                'layers': [{'height': 2},
                           {'height': 2},
                           {'height': 2},
                           {'height': 1, 'activations': activations.linears}],
                'learning_rate': l_rate,
                'momentum': 0.6}
    leaky   = {'name': 'Leaky ReLu',
                'layers': [{'height': 2},
                           {'height': 8, 'activations': activations.leaky_relus},
                           {'height': 8, 'activations': activations.leaky_relus},
                           {'height': 1, 'activations': activations.linears}],
                'learning_rate': l_rate,
                'momentum': 0.6}
    relu    = {'name': 'ReLu',
                'layers': [{'height': 2},
                           {'height': 8, 'activations': activations.relus},
                           {'height': 8, 'activations': activations.relus},
                           {'height': 1, 'activations': activations.linears}],
                'learning_rate': l_rate,
                'momentum': 0.6}

    neural_models = []
    for model in [sigmoid, leaky, relu]:
        neural_models.append(neural_model.Network(**model))

    ridge_model = [linear_models.RegularisedLinearRegression(
        name = 'Ridge',
        beta_func = linear_models.beta_ridge,
        momentum = 0.6,
        _lambda = 0.001,
        x_shape = X_train.shape[1],
        learning_rate = 0.01
        ), X_train, X_validate, data['y_train'], data['y_validate']]

    subplots = [([*neural_models, ridge_model], {'mini_batch_size': mini_batch_size})]

    errors, subtitle, subplots, metrics_string = sgd.sgd_on_models(data['x_train'], data['x_validate'], data['y_train'], data['y_validate'], *subplots, epochs=epochs, epochs_without_progress=epochs_without_progress)

    title = ['Best models of each type, regression', 'regression_vs', subtitle]

    sgd.plot_sgd_errors(errors, title, metrics_string)

    print("Model performances on test data")
    for model in neural_models:
        y_pred = model.predict(data['x_test'])
        mse = metrics.MSE(data['y_test'], y_pred)
        print(f"{model.name}: {mse}")
    ridge_model[0].predict(X_test)
    mse = metrics.MSE(data['y_test'], y_pred)
    print(f"{ridge_model[0].name}: {mse}")

def mnist_classification(data, epochs=10000, epochs_without_progress=2000, mini_batch_size=40):
    """Plots and writes out performance of neural and logistic models on classification problem for the MNIST dataset"""
    total_steps = epochs * len(data['x_train'])//mini_batch_size

    learning_rates = [
            learning_rate.Learning_rate(base=base, decay=decay).ramp_up(10).compile(total_steps)
            for base, decay in [(3e-3, 2.5e-5), (2e-3, 5e-5)]]

    neural =   {'layers': [{'height': 64},
                           {'height': 32},
                           {'height': 10, 'activations': activations.sigmoids, 'd_func': lambda a, y, _: y - a}],
                'learning_rate': learning_rates[0],
                'momentum': 0.6}

    logistic = {'name': 'Logistic',
                'layers': [{'height': 64},
                           {'height': 10, 'activations': activations.softmaxs, 'd_func': lambda a, y, _: y - a}],
                'learning_rate': learning_rates[1],
                'momentum': 0.6}

    neural_model_ = neural_model.Network(**neural)
    logistic_model = neural_model.Network(**logistic)

    subplots = [([neural_model_, logistic_model], {'mini_batch_size': mini_batch_size})]

    sgd_on_models_kwargs = {
        'epochs': epochs,
        'epochs_without_progress': epochs_without_progress,
        'metric': metrics.accuracy_loss
    }
    data_args = [data['x_train'], data['x_validate'], data['y_train'], data['y_validate']]
    errors, subtitle, subplots, metric_string = sgd.sgd_on_models(*data_args, *subplots, **sgd_on_models_kwargs)

    title = ['Neural model, Classification test', 'neural_classification', subtitle]

    sgd.plot_sgd_errors(errors, title, metric_string)

    classification_subplots = [(subplots[0][0], subplots[1]), (subplots[0][1], subplots[1])]

    helpers.classification_accuracy(classification_subplots, data)

def mnist_softmax_sigmoid(data, epochs=5000, epochs_without_progress=2000, mini_batch_size=40):
    """Compares softmax and sigmoid as activation of final layer on classification problem for the MNIST dataset

    We have had some issues making softmax working for several layers, and we can't figure out exactly why.
    The plot generated here does however show what goes wrong, namely exploding weights in the hidden layer."""

    total_steps = epochs * len(data['x_train'])//mini_batch_size

    l_rate = learning_rate.Learning_rate(base=3e-3, decay=2.5e-5).ramp_up(10).compile(total_steps)

    common_kwargs = {'momentum': 0.6, 'learning_rate': l_rate}
    subsubplot_uniques = [{'name': 'Softmax', 'layers': [{'height': 64}, {'height': 32}, {'height': 10, 'activations': activations.softmaxs, 'd_func': lambda a, y, _: y - a}]},
                       {'name': 'Sigmoid', 'layers': [{'height': 64}, {'height': 32}, {'height': 10, 'd_func': lambda a, y, _: y - a}]}]

    subplot_uniques = [{}]

    unique_sgd_kwargs = [{'mini_batch_size': mini_batch_size}]

    neural_models = helpers.make_models(
        neural_model.Network,
        common_kwargs,
        subplot_uniques,
        len(unique_sgd_kwargs),
        subsubplot_uniques
        )

    subplots = [(models, sgd_kwargs) for models, sgd_kwargs in zip(neural_models, unique_sgd_kwargs*len(neural_models))]

    sgd_on_models_kwargs = {
        'epochs': epochs,
        'epochs_without_progress': epochs_without_progress,
        'metric': metrics.accuracy_loss
    }
    data_args = [data['x_train'], data['x_validate'], data['y_train'], data['y_validate']]
    errors, subtitle, subplots, metric_string = sgd.sgd_on_models(*data_args, *subplots, **sgd_on_models_kwargs)

    title = ['Neural model, softmax vs sigmoid', 'softmax_vs_sigmoid', subtitle]

    sgd.plot_sgd_errors(errors, title, metric_string)

    trained_models = [model for models, _ in subplots for model in models]
    for model in trained_models:
        model.plot_history()

def tune_neural_reg(data, epochs=10000, epochs_without_progress=500, mini_batch_size=20):
    """Makes a heatmap of performance of different neural models on regrression on real terrain"""
    total_steps =  epochs * len(data['x_train'])//mini_batch_size
    learning_rates = [
            learning_rate.Learning_rate(base=base, decay=decay).ramp_up(10).compile(total_steps)
            for base, decay in [(1e-3, 1/25000), (1e-2, 1/20000), (3e-2, 1/20000)]]

    common_kwargs = {'momentum': 0.6}
    subsubplot_uniques = [{'learning_rate': learning_rate_func} for learning_rate_func in learning_rates]

    unique_sgd_kwargs = [{'mini_batch_size': mini_batch_size}]

    activation_names = ['ReLu', 'Leaky ReLu', 'Sigmoid']
    _activations = [activations.relus, activations.leaky_relus, activations.sigmoids]
    for _activations, activation_name in zip(_activations, activation_names):
        print(f"Testing for activation function {activation_name}")
        heights = [2, 4, 6, 8]
        hiddens_1 = [[{'height': hidden, 'activations': _activations}] for hidden in heights]
        hiddens_2 = [[{'height': hidden, 'activations': _activations},
                      {'height': hidden, 'activations': _activations}] for hidden in heights]
        subplot_uniques = [{'layers': [{'height': 2}, *hidden_layers, {'height': 1, 'activations': activations.linears}]}
                           for hidden_layers in hiddens_1 + hiddens_2]
        
        neural_models = helpers.make_models(
            neural_model.Network,
            common_kwargs,
            subplot_uniques,
            len(unique_sgd_kwargs),
            subsubplot_uniques
            )

        models = [model for models in neural_models for model in models]

        tune_obj = tune.Tune(models, data, [metrics.MSE], name=f"Neural reg tune {activation_name}")
        tune_obj.validate(epochs=epochs, mini_batch_size=mini_batch_size, epochs_without_progress=epochs_without_progress)
        tune_obj.plot_validation_errors()

def tune_mnist_classification(data, epochs=15000, epochs_without_progress=500, mini_batch_size=40):
    """Makes a heatmap of performance of neural and logistic models on classification on the MNIST dataset"""
    total_steps =  epochs * len(data['x_train'])//mini_batch_size
    learning_rates = [
            learning_rate.Learning_rate(base=base, decay=decay).ramp_up(1000).compile(total_steps)
            for base, decay in [(3e-3, 1/40000), (2e-3, 1/20000), (5e-3, 1/10000)]]

    common_kwargs = {'momentum': 0.6}
    subsubplot_uniques = [{'learning_rate': learning_rate_func} for learning_rate_func in learning_rates]
    unique_sgd_kwargs = [{'mini_batch_size': mini_batch_size}]

    heights = [8, 16, 32, 64]
    hiddens_1 = [[{'height': hidden}] for hidden in heights]
    hiddens_2 = [[{'height': hidden}, {'height': hidden}] for hidden in heights]

    subplot_uniques = [{'layers': [{'height': 64}, *hidden_layers, {'height': 10, 'd_func': lambda a, y, _: y - a}]}
                      for hidden_layers in hiddens_1 + hiddens_2]

    neural_models = helpers.make_models(
        neural_model.Network,
        common_kwargs,
        subplot_uniques,
        len(unique_sgd_kwargs),
        subsubplot_uniques
        )

    models = [model for models in neural_models for model in models]

    neural_tune = tune.Tune(models, data, [metrics.accuracy_loss], name="MNIST class tune", polynomials=None)
    neural_tune.validate(epochs=epochs, mini_batch_size=mini_batch_size, epochs_without_progress=epochs_without_progress)
    neural_tune.plot_validation_errors()

if __name__ == '__main__':
    import real_terrain
    import mnist

    terrain_data = real_terrain.get_data(20)
    mnist_data = mnist.get_data(0.6, 0.2)
    
    functions = {
        'conf': (conf_interval_plot, terrain_data),
        'momentum': (momentum_plot, terrain_data),
        'beta_variance': (beta_variance, terrain_data),
        'neural_reg_tune': (tune_neural_reg, terrain_data),
        'reg': (neural_regression, terrain_data),
        'mnist': (mnist_classification, mnist_data),
        'mnist_tune': (tune_mnist_classification, mnist_data),
        'softmax_issue': (mnist_softmax_sigmoid, mnist_data)
    }

    if 'all' in sys.argv:
        for name, (f, data) in functions.items():
            print(f"Performing {name}...")
            f(data)
    else:
        for arg in sys.argv[1:]:
            if arg in functions:
                f, data = functions[arg]
                print(f"Performing {arg}...")
                f(data)
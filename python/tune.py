import sys
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import linear_models
import resampling
import metrics
import plotting
import sgd
import helpers

np.random.seed(10)

class Tune:
    """Validates, finds optimal hyperparameters, and tests them.

    Parameters:
    -----------
    models:     list[object]
                Instances of classes with a scaler, a fit method and a predict method.
    data:       dict
                All training and testing data, as constructed by data_handling.make_data_dict,
    poly_iter:  iterable
                All maximum polynomials to validate for.
    para_iters: list[iterable]
                As many sets of hyperparameters as there are models.
    name:       str
                Name of tuning run. Filenames of plots and datasets are prepended by the name.
    verbose:    int
                How much to print out during validation, with 0 being nothing, and 2 being evrything.
    """
    def __init__(self, models, data, _metrics, polynomials=8, name="Tune", verbose=1):
        self.models = models
        self.verbose = verbose
        self.data = data
        self.polynomials = polynomials
        self._metrics = _metrics
        self.name = name

        for metric in _metrics:
            assert metric.__doc__ is not None, "Metrics need docstrings"

    def validate(self, epochs=200, **kwargs):
        """Validates all models for all polynomials and all parameters, and stores data in validation_errors.

        Creates and populates pandas.DataFrame validation_errors with MSE from bootstrap and kfold resampling techniques,
        as well as model bias and variance from bootstrap, for all combinations of hyperparameters.

        Parameters:
        -----------
        epochs:     int
                    Number of epochs. 200 by default.
        **kwargs:   keyword arguments
                    Passed to sgd.sgd
        """
        model_properties = [model.property_dict for model in self.models]
        model_uniques, model_common = helpers.filter_dicts(model_properties)

        for unique in model_uniques:
            assert len(unique) == len(model_uniques[0]), "All models must have the same property types"
            assert len(unique) > 0, "Two models with the same property_dict has been sent in"

        index_parameters = helpers.listify_dict_values(model_uniques)
        parameter_names = [key for key in index_parameters]
        model_parameters = [values for _, values in index_parameters.items()]
        metric_texts = [metric.__doc__ for metric in self._metrics]

        errors_index = pd.MultiIndex.from_product([metric_texts, *model_parameters], names = ['Metric', *parameter_names])
        self.errors_df = pd.DataFrame(dtype=float, index=errors_index, columns=range(1, epochs+1))

        if self.polynomials is not None:
            X_train = linear_models.poly_design_matrix(self.polynomials, self.data['x_train'])
            X_validate = linear_models.poly_design_matrix(self.polynomials, self.data['x_validate'])

        y_train, y_validate = self.data['y_train'], self.data['y_validate']

        idx = pd.IndexSlice

        for i, (model, model_unique) in enumerate(zip(self.models, model_uniques)):
            print(f"\r |{'='*(i*50//len(self.models))}{' '*(50-i*50//len(self.models))}| {i/len(self.models):.2%}", end="", flush=True)
            
            if model.name == 'OLS' or model.name == 'Ridge':
                x_train, x_validate = X_train, X_validate
            else:
                x_train, x_validate = self.data['x_train'], self.data['x_validate']

            model_indexes = [model_unique[key] for key in index_parameters]

            for j, metric in enumerate(self._metrics):
                model.compile()
                errors = sgd.sgd(model, x_train, x_validate, y_train, y_validate, epochs=epochs, metric=metric, **kwargs)[1]
                self.errors_df.loc[tuple(pd.IndexSlice[s] for s in [metric.__doc__, *model_indexes])] = errors

        print("")

        self.errors_df.dropna(thresh=2, inplace=True)
        self.errors_df.to_csv("../dataframes/tune.csv")


    def plot_validation_errors(self):
        """Plots heatmaps of validation errors for last epoch.

        Makes one plot for each combination of two factors, for each metric."""
        index = self.errors_df.index
        names = index.names

        for metric in self._metrics:
            for i in range(1, len(names)):
                for j in range(i+1, len(names)):
                    x_label = names[j]
                    y_label = names[i]

                    heatmap_df = self.errors_df.loc[metric.__doc__].iloc(axis=1)[-1].copy()
                    remove_levels = np.ones(len(names), dtype=bool)
                    remove_levels[np.array([0,i,j])] = False
                    print(remove_levels == True)
                    print(names)
                    if np.nonzero(remove_levels == True)[0] > 0:
                        for k in np.nonzero(remove_levels == True):
                            heatmap_df = heatmap_df.mean(level=k)

                    heatmap_values = heatmap_df.values

                    heatmap_df = self.errors_df.loc[metric.__doc__].iloc(axis=1)[-1].groupby(level=[names[i], names[j]]).mean()
                    x_ticks = heatmap_df.index.get_level_values(x_label)
                    y_ticks = heatmap_df.index.get_level_values(y_label)
                    heatmap_values = heatmap_df.unstack().values*10000

                    f, ax = plt.subplots(figsize=(9, 6))

                    filename = f"{self.name.replace(' ', '_').replace(',', '')}_{names[i]}_vs_{names[j]}_{metric.__doc__}"
                    title = f"{self.name}: {names[i]} vs {names[j]} ({metric.__doc__}, scaled $10^{-4}$)"

                    plt.title(title)

                    min_y, min_x = np.nonzero(heatmap_values == np.amin(heatmap_values))
                    for i in range(len(min_y)):
                        ax.add_patch(Rectangle((min_x[i], min_y[i]), 1, 1, fill=False, edgecolor='pink', lw=3))
                    ax = sns.heatmap(heatmap_values,
                                     cbar=False,
                                     annot=True,
                                     square=True,
                                     yticklabels=np.unique(y_ticks),
                                     xticklabels=np.unique(x_ticks),
                                     fmt='3.0f')
                    plt.yticks(rotation=45)
                    plt.ylabel(y_label)
                    plt.xlabel(x_label)
                    plt.savefig(f"../plots/{filename}")
                    plt.show()


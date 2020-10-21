import sys
import numpy as np
import pandas as pd

import linear_models
import resampling
import metrics
import plotting

np.random.seed(10)

def poly_design_matrix(p, x):
    """Make a design matrix where each column is a combination of the input x's data columns to powers up to p.

    Parameters:
    -----------
    p:          int
                Maximum polynomial degree of columns.
    x:          array of shape (n, f)
                Predictor variable, with each row being one datapoint.

    Returns:
    --------
    X:          2-dimensional array
                Design matrix with rows as data points and columns as features.
    """
    powers = np.arange(p+1)[np.newaxis,:].repeat(x.shape[1], axis=0)
    # Make a (p+1)**x.shape[0] x 2 matrix with pairs of power combinations
    pow_combs = np.array(np.meshgrid(*powers)).T.reshape(-1, x.shape[1])
    # Remove all combinations with combined powers of over p
    pow_combs = pow_combs[np.sum(pow_combs, axis=1) <= p]
    # Make third order tensor where last dimension is used for each input dimension to its appropriate power
    X_powers = x[:,:,np.newaxis].astype(float).repeat(p+1, axis=2)**powers
    # Multiply the powers of X together according to pow_combs
    X = X_powers[:,0,pow_combs[:,0]] * X_powers[:,1,pow_combs[:,1]]
    return X

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
    def __init__(self, models, data, poly_iter, para_iters, name="", verbose=1):
        self.models = models
        self.poly_iter = poly_iter
        self.para_iters = para_iters
        self.verbose = verbose
        self.data = data

    def validate(self, test_as_well=False):
        """Validates all models for all polynomials and all parameters, and stores data in validation_errors.

        Creates and populates pandas.DataFrame validation_errors with MSE from bootstrap and kfold resampling techniques,
        as well as model bias and variance from bootstrap, for all combinations of hyperparameters.

        Parameters:
        -----------
        test_as_well:
                    bool
                    If True, test all models on test set as well, and store results
        """
        # Make dataframe for validation data
        lambda_list = np.unique(np.concatenate(self.para_iters)) if len(self.para_iters) > 1 else self.para_iters[0]
        validation_index = pd.MultiIndex.from_product([
            ['Boot MSE', 'Boot Bias', 'Boot Var', 'Kfold MSE', 'Test MSE'],
            [model.name for model in self.models],
            [_lambda for _lambda in lambda_list]],
            names = ['Metric', 'Model', 'Lambda']
            )
        self.validation_errors = pd.DataFrame(dtype=float, index=validation_index, columns=self.poly_iter)
        self.validation_errors.sort_index(inplace=True)

        for i, p in enumerate(self.poly_iter):
            if self.verbose >= 1:
                print(f"{(i+1):2d}/{len(poly_iter)}: Polynomial of degree {p}")

            X_train_validate = poly_design_matrix(p, self.data['x_train_validate'])
            X_train, X_validate = np.split(X_train_validate, [self.data['x_train'].shape[0]])

            for j, (model, para_iter) in enumerate(zip(self.models, self.para_iters)):
                if self.verbose >= 1:
                    print(f"    {(j+1):2d}/{len(self.models)}: Model: {model.name}")

                for k, _lambda in enumerate(para_iter):
                    if self.verbose >= 2:
                        print(f"        {(k+1):2d}/{len(para_iter)}: Lambda = {_lambda}")

                    model.set_lambda(_lambda)
                    boot_mse, boot_bias, boot_var = resampling.bootstrap(model, X_train, X_validate, self.data['y_train'], self.data['y_validate'], R=50)
                    kfold_mse = resampling.kfold(model, X_train_validate, self.data['y_train_validate'], n_folds=10)

                    self.validation_errors.loc['Boot MSE', model.name, _lambda][p] = boot_mse
                    self.validation_errors.loc['Boot Bias', model.name, _lambda][p] = boot_bias
                    self.validation_errors.loc['Boot Var', model.name, _lambda][p] = boot_var
                    self.validation_errors.loc['Kfold MSE', model.name, _lambda][p] = kfold_mse

                    if test_as_well:
                        X_train_c = np.copy(X_train)

                        model.scaler.fit(X_train_c[:,1:])
                        X_train_c[:,1:] = model.scaler.transform(X_train_c[:,1:])
                        model.set_lambda(_lambda)
                        model.fit(X_train_c, self.data['y_train'])

                        X_test = poly_design_matrix(p, self.data['x_test'])
                        X_test[:,1:] = model.scaler.transform(X_test[:,1:])
                        y_pred = model.predict(X_test)

                        test_mse = metrics.MSE(data['y_test'], y_pred)
                        self.validation_errors.loc['Test MSE', model.name, _lambda][p] = test_mse

                    #if self.verbose >= 2:
                        #print(f"         error: {(boot_mse+kfold_mse)/2}")

        self.validation_errors.dropna(thresh=2, inplace=True)
        #self.validation_errors.drop_duplicates(inplace=True)

    def optimal_model_search(self):
        """Uses validation_errors to find the best models.

        Assumes pandas.DataFrame validation_errors has already been created by validate method, and makes a new dataframe.
        optimal_model stores the absolute best combination of max polynomial and lambda for each model.
        """
        optimal_model_index = pd.MultiIndex.from_product([
            [model.name for model in self.models],
            ['Bootstrap', 'Kfold', 'Average']],
            names= ['Model', 'Resampling technique']
            )
        self.optimal_model = pd.DataFrame(index=optimal_model_index, columns=['Lambda', 'Polynomial', 'Validation error', 'Test error'])

        average_MSE = (self.validation_errors.loc['Boot MSE'] + self.validation_errors.loc['Kfold MSE'])/2
        
        for model in self.models:
            self.optimal_model.loc[model.name, 'Bootstrap']['Lambda', 'Polynomial'] = self.validation_errors.loc['Boot MSE', model.name].stack().idxmin()
            self.optimal_model.loc[model.name, 'Kfold']['Lambda', 'Polynomial'] = self.validation_errors.loc['Kfold MSE', model.name].stack().idxmin()
            self.optimal_model.loc[model.name, 'Average']['Lambda', 'Polynomial'] = average_MSE.loc[model.name].stack().idxmin()
            
            self.optimal_model.loc[model.name, 'Bootstrap']['Validation error'] = self.validation_errors.loc['Boot MSE', model.name].stack().min()
            self.optimal_model.loc[model.name, 'Kfold']['Validation error'] = self.validation_errors.loc['Kfold MSE', model.name].stack().min()
            self.optimal_model.loc[model.name, 'Average']['Validation error'] = average_MSE.loc[model.name].stack().min()

    def test(self):
        """Tests out the models that validate and optimal_model_search has determined are best.

        Uses the pandas.DataFrame optimal_model created by optimal_model_search method, and tests each of the models
        on testing data. Plots the result side by side with the actual dependent variable.

        """
        for (model_name, technique), (_lambda, poly, _, _) in self.optimal_model.iterrows():

            for _model in self.models:
                if _model.name == model_name:
                    model = _model

            X = poly_design_matrix(int(poly), self.data['x_train'])

            model.scaler.fit(X[:,1:])
            X[:,1:] = model.scaler.transform(X[:,1:])
            model.set_lambda(_lambda)
            model.fit(X, self.data['y_train'])

            X_test = poly_design_matrix(int(poly), self.data['x_test'])
            X_test[:,1:] = model.scaler.transform(X_test[:,1:])
            y_pred = model.predict(X_test)

            mse = metrics.MSE(data['y_test'], y_pred)
            self.optimal_model.loc[model_name, technique]['Test error'] = mse

            if self.verbose >= 1:
                print(f"MSE: {mse} for poly = {poly} and lambda = {_lambda} with model {model_name}.")
            
            plotting.side_by_side(
                ['Predicted', self.data['x_test'], y_pred],
                ['Ground truth', self.data['x_test'], self.data['y_test']],
                title=f"Test: {model.name}, $p$={poly}, $\\lambda$={_lambda}, best according to {technique}",
                filename=f"{name}_test_{model.name}_p{poly}{[f'_lambda{_lambda}', ''][int(_lambda==0)]}")

    def save(self):
        """Saves dataframes created by .validation and .optimal_model_search to csv files.
        """
        if 'validation_errors' in self.__dict__:
            self.validation_errors.to_csv(f'../dataframe/{name}_validation_errors.csv')
        if 'optimal_per_poly' in self.__dict__:
            self.optimal_per_poly.to_csv(f'../dataframe/{name}_optimal_per_poly.csv')
        if 'optimal_model' in self.__dict__:
            self.optimal_model.to_csv(f'../dataframe/{name}_optimal_model.csv')

            header = ["Lambda", "Poly", "Validation", "Test"]
            index = ["Model", "Technique"]
            self.optimal_model.to_latex(f'../dataframe/{name}_optimal_model.tex', header=header, index=index, multirow=True, float_format="{:0.3e}".format)



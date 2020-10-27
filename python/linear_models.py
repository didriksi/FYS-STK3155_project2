import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

import metrics

def beta_ridge(_lambda, X, y):
    """Returns a beta with parameters fitted for some X and y, and _lambda hyperparameter. Method is Ridge.

    Parameters:
    -----------
    _lambda:    float
                Regularisation parameter.
    X:          2-dimensional array
                Design matrix with rows as data points and columns as features.
    y:          1-dimensional array
                Dependent variable.
    
    Returns:
    --------
    beta:       array
                Array of parameters
    """
    U, s, VT = np.linalg.svd(X.T @ X + _lambda * np.eye(X.shape[1]))
    D = np.zeros((U.shape[0], VT.shape[0])) + np.eye(VT.shape[0]) * np.append(s, np.zeros(VT.shape[0] - s.size))
    invD = np.linalg.inv(D)
    invTerm = VT.T @ np.linalg.inv(D) @ U.T
    beta = invTerm @ X.T @ y
    return beta

def beta_lasso(_lambda, X, y):
    """Returns a beta with parameters fitted for some X and y, and _lambda hyperparameter. Method is LASSO.

    Parameters:
    -----------
    _lambda:    float
                Regularisation parameter.
    X:          2-dimensional array
                Design matrix with rows as data points and columns as features.
    y:          1-dimensional array
                Dependent variable.
    
    Returns:
    --------
    beta:       array
                Array of parameters
    """
    lasso = Lasso(alpha=_lambda, fit_intercept=False, max_iter=200000)
    lasso.fit(X, y)
    return lasso.coef_

class LinearRegression:
    """Fits on data, and makes some predictions based on linear regression.

    Parameters:
    -----------
    name:       str
                Name of method. "OLS" by default, used by subclasses.
    scaler:     object
                Instance of class that has a fit and transform method for scaling predictor data.
    """
    def __init__(self, name="OLS", scaler=StandardScaler()):
        self.name = name
        self.scaler = scaler
        self._lambda = 0

    def fit(self, X, y):
        """Fit a beta array of parameters to some predictor and dependent variable

        Parameters:
        -----------
        X:          2-dimensional array
                    Design matrix with rows as data points and columns as features.
        y:          1-dimensional array
                    Dependent variable.
        """
        self.beta = np.linalg.pinv(X) @ y

    def set_lambda(self, _lambda):
        """Does nothing. Only here for compatibility with subclasses that have a lambda parameter.
        """
        pass

    def predict(self, X):
        """Predicts new dependent variable based on beta from .fit method, and a new design matrix X.

        Parameters:
        -----------
        X:          2-dimensional array
                    Design matrix with rows as data points and columns as features.
        
        Returns:
        --------
        y_pred:     1-dimensional array
                    Predicted dependent variable.
        """
        y_pred = X @ self.beta
        return y_pred

    def compile(self, beta_shape, learning_rate=0, momentum=0, beta_initialiser=lambda shape: np.random.randn(*shape)):
        self.momentum = momentum
        self.beta = beta_initialiser(beta_shape)
        self.velocity = np.zeros_like(self.beta)
        self.step = 0
        self.set_learning_rate(learning_rate)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate if callable(learning_rate) else lambda step: learning_rate

    def cost_diff(self, X, n, y, y_tilde):
        return np.dot(X.T, 2/n*(y_tilde - y[:,np.newaxis])) + 2*self._lambda*self.beta #TODO: Fix shape

    def update_parameters(self, X, n, y, y_tilde):
        self.velocity = self.velocity*self.momentum + self.learning_rate(self.step)*self.cost_diff(self, X, n, y, y_tilde)
        self.beta -= self.velocity
        self.step += 1

    def conf_interval_beta(self, y, y_pred, X):
        """Estimates the 99% confidence interval for array of parameters beta.

        Parameters:
        -----------
        y:          1-dimensional array
                    Ground truth dependent variable.
        y_pred:     1-dimensional array
                    Predicted dependent variable.
        X:          2-dimensional array
                    Design matrix with rows as data points and columns as features.
        
        Returns:
        --------
        confidence_interval:
                    list(array, array)
                    Lowest and highest possible values for beta, with 99% confidence.
        confidence_deviation:
                    array
                    Deviation from value in confidence interval.
        """
        sigma2_y = metrics.MSE(y, y_pred)
        sigma_beta = np.sqrt((sigma2_y * np.linalg.inv(X.T @ X)).diagonal())
        confidence_interval = np.array([self.beta - 2*sigma_beta, self.beta + 2*sigma_beta])
        return confidence_interval, 2*sigma_beta

class RegularisedLinearRegression(LinearRegression):
    """Fits on data, and makes some predictions based on regularised linear regression.

    Parameters:
    -----------
    name:       str
                Name of method. "OLS" by default, used by subclasses.
    beta_func:  function
                Function used for fitting beta. Has to be able to take _lambda, X and y, and return an array of parameters beta.
    scaler:     object
                Instance of class that has a fit and transform method for scaling predictor data.
    """
    def __init__(self, name, beta_func, scaler=StandardScaler()):
        super().__init__(name, scaler)
        self.beta_func = beta_func

    def set_lambda(self, _lambda):
        """Sets a specific parameter value for the beta_func.

        Parameters:
        -----------
        _lambda:    float
                    Regularisation parameter.
        """
        self._lambda = _lambda

    def fit(self, X, y):
        """Fit a beta array of parameters to some predictor and dependent variable

        Parameters:
        -----------
        X:          2-dimensional array
                    Design matrix with rows as data points and columns as features.
        y:          1-dimensional array
                    Dependent variable.
        """
        self.beta = self.beta_func(self._lambda, X, y)

    def cost_diff(self, X, n, y, y_tilde):
        return super().cost_diff(X, n, y, y_tilde) + 2*self._lambda*self.beta

    def conf_interval_beta(self, y, y_pred, X):
        """Does nothing. Only here to give an error if someone tries to call it, because its super class has one that works.
        """
        raise NotImplementedError(f'Can only find confidence interval of beta from OLS, not {name}')
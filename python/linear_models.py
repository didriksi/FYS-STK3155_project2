import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

import metrics

def beta_initialiser(shape):
    """$3 \\cdot \\mathbb{N}(0,1)$"""
    return 3*np.random.randn(*shape)

class LinearRegression:
    """Fits on data, and makes some predictions based on linear regression.

    Parameters:
    -----------
    name:       str
                Name of method. "OLS" by default, used by subclasses.
    scaler:     object
                Instance of class that has a fit and transform method for scaling predictor data.
    """
    def __init__(self, name="OLS", scaler=StandardScaler(), x_shape=1, init_conds=1, learning_rate=0.01, momentum=0, beta_initialiser=beta_initialiser):
        self.name = name
        self.scaler = scaler
        self.momentum = momentum
        self.beta = beta_initialiser((x_shape, init_conds))
        self.beta_initialiser = beta_initialiser
        self.velocity = np.zeros_like(self.beta)
        self.step = 0
        self.learning_rate = learning_rate if callable(learning_rate) else lambda step: learning_rate

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
        return self.beta

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

    def compile(self, x_shape=1, init_conds=1):
        """Prepares the model for training by gradient descent.
        
        Parameters:
        -----------
        beta_shape: tuple of ints
                    Desired shape of beta.
        learning_rate:
                    (int -> float) or float
                    Either function that takes in step number and returns float, or just a float.
        momentum:   float
                    Adds momentum to gradient descent. Default is 0.
        beta_initialiser:
                    (tuple of ints -> array of shape of tuples)
                    Function that returns initial beta with shape decided by beta_shape. Default is standard normal.
        """
        self.beta = self.beta_initialiser(self.beta.shape)
        self.velocity = np.zeros_like(self.beta)
        self.step = 0

    def get_gradient(self, X, y, y_tilde):
        """Differentiates MSE for given data.
        
        Parameters:
        -----------
        X:          2-dimensional array
                    Design matrix with rows as data points and columns as features.
        y:          1-dimensional array
                    Actual dependent variable.
        y_tilde:    1-dimensional array
                    Predicted dependent variable.

        Returns:
        --------
        gradient:   array
                    Differentiated MSE.

        """
        return np.dot(X.T, 2/X.shape[0]*(y_tilde - y))

    def update_parameters(self, X, y, y_tilde):
        """Performs one step of gradient descent.

        Parameters:
        -----------
        X:          2-dimensional array
                    Design matrix with rows as data points and columns as features.
        y:          1-dimensional array
                    Actual dependent variable.
        y_tilde:    1-dimensional array
                    Predicted dependent variable.
        """
        self.velocity = self.velocity*self.momentum + self.learning_rate(self.step)*self.get_gradient(X, y, y_tilde)
        self.beta -= self.velocity
        self.step += 1

    @property
    def property_dict(self):
        properties = {'model_name': self.name, 'momentum': self.momentum, 'learning_rate': self.learning_rate_name}
        if self.beta_initialiser.__doc__ is not None:
            properties['Init $\\beta$'] = self.beta_initialiser.__doc__
        return properties

    @property
    def parallell_runs(self):
        return self.beta.shape[1]

    @property
    def learning_rate_name(self):
        if callable(self.learning_rate):
            if self.learning_rate.__doc__ is not None:
                return self.learning_rate.__doc__
            else:
                return f"(0): {self.learning_rate(0)}"
        else:
            return f": flat {self.learning_rate}"

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
    def __init__(self, name, beta_func, _lambda=0.01, scaler=StandardScaler(), x_shape=1, init_conds=1, learning_rate=0.01, momentum=0, beta_initialiser=beta_initialiser):
        super().__init__(name=name, scaler=scaler, x_shape=x_shape, init_conds=init_conds, learning_rate=learning_rate, momentum=momentum, beta_initialiser=beta_initialiser)
        self.beta_func = beta_func
        self._lambda = _lambda

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
        return self.beta

    def get_gradient(self, X, y, y_tilde):
        """Differentiates MSE for given data.
        
        Parameters:
        -----------
        X:          2-dimensional array
                    Design matrix with rows as data points and columns as features.
        y:          1-dimensional array
                    Actual dependent variable.
        y_tilde:    1-dimensional array
                    Predicted dependent variable.

        Returns:
        --------
        gradient:   array
                    Differentiated MSE.
        """
        return super().get_gradient(X, y, y_tilde) + 2*self._lambda*self.beta

    @property
    def property_dict(self):
        properties = super().property_dict
        properties['_lambda'] = self._lambda
        return properties

    def conf_interval_beta(self, y, y_pred, X):
        """Does nothing. Only here to give an error if someone tries to call it, because its super class has one that works.
        """
        raise NotImplementedError(f'Can only find confidence interval of beta from OLS, not {name}')

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


























import numpy as np

import metrics

def bootstrap(model, X_train, X_validate, y_train, y_validate, R=50):
    """Resampling techique that draws with replacement from training set, and validates how well a model fitted on that set does.

    Parameters:
    -----------
    model:      object
                Instance of class with a scaler, a fit method and a predict method.
    X_train:    2-dimensional array
                Design matrix with rows as data points and columns as features, and training data.
    X_validate: 2-dimensional array
                Design matrix with rows as data points and columns as features, and validation data.
    y_train:    1-dimensional array
                Dependent variable, training data.
    y_validate: 1-dimensional array
                Dependent variable, validate data.
    R:          int
                Number of resamples.

    Returns:
    --------
    mse:        int
                Mean squared error from validation.
    bias:       int
                Bias for models fitted on the different samples of training data.
    var:        int
                Variation for models fitted on the different samples of training data.
    """
    n = X_train.shape[0]
    y_boot_tilde = np.empty((R, y_validate.shape[0]))

    for r in range(R):
        boot_mask = np.random.randint(0, n, n)
        X_boot = X_train[boot_mask,:]
        y_boot = y_train[boot_mask]

        model.scaler.fit(X_boot[:,1:])
        X_boot[:,1:] = model.scaler.transform(X_boot[:,1:])
        model.fit(X_boot, y_boot)

        X_validate[:,1:] = model.scaler.transform(X_validate[:,1:])
        y_boot_tilde[r] = model.predict(X_validate)

    mse = metrics.MSE(y_validate, y_boot_tilde)
    bias = metrics.model_bias(y_validate, y_boot_tilde)
    var = metrics.model_variance(y_boot_tilde)

    return mse, bias, var

def kfold(model, X, y, n_folds=5):
    """Resampling techique that draws with replacement from training set, and validates how well a model fitted on that set does.

    Parameters:
    -----------
    model:      object
                Instance of class with a scaler, a fit method and a predict method.
    X:          2-dimensional array
                Design matrix with rows as data points and columns as features, and both training and validation data.
                It will use random parts of it, but only so much that it trains on half of it.
    y:          1-dimensional array
                Dependent variable, both training and validation data.
    n_folds:    int
                Number of folds the data is divided into.

    Returns:
    
    mse:        int
                Mean squared error from validation.
    """
    datasize = X.shape[0]//n_folds
    foldsize = datasize//n_folds

    shuffled_indexes = np.arange(datasize, dtype=np.int32)
    np.random.shuffle(shuffled_indexes)

    # To make sure bootstrap and kfold get about as much data in training set, and is shuffled
    X_shuffled = X[shuffled_indexes]
    y_shuffled = y[shuffled_indexes]

    mse = np.empty(n_folds)
    for k in range(n_folds):
        k_validate = np.s_[int(k*foldsize):int((k+1)*foldsize)]
        indexes = np.arange(datasize)
        k_train = np.s_[np.logical_or(indexes < k*foldsize, (k+1)*foldsize < indexes)]

        model.scaler.fit(X_shuffled[k_train,1:])
        X_shuffled[:,1:] = model.scaler.transform(X_shuffled[:,1:])

        model.fit(X_shuffled[k_train], y_shuffled[k_train])
        y_pred = model.predict(X_shuffled[k_validate])

        mse[k] = metrics.MSE(y_shuffled[k_validate], y_pred)

    return np.mean(mse)
import numpy as np
import matplotlib.pyplot as plt

import tune
import metrics
import linear_models

def sgd(data, mini_batches, epochs, learning_rate, polynomials, momentum, init_beta=lambda b: np.random.randn(b), metric=metrics.MSE):
    """Stochastic gradient descent

    Parameters:
    -----------
    data:       dict
                All training and testing data, as constructed by data_handling.make_data_dict.
    mini_batches:
                int
                Number of mini-batches.
    epochs:     int
                Number of epochs.
    learning_rate:
                int -> float
                Function that takes in a timestep, and returns a learning rate.
    polynomials:int
                Maximum degree of polynomials in design matrix
    momentum:   float
                Gamma parameter, determining how much momentum to keep. Set to zero for standard SGD.
    init_beta:  int b -> array of shape (b, )
                Function that takes in the amount of beta parameters, and returns an initial beta array.
    metric:     array of shape(n, ), array of shape(n, ) -> float
                Should take in y and y_tilde, and return the total error.

    
    Returns:
    --------
    
    """

    data_size = data['x_train'].shape[0]
    assert data_size%mini_batches == 0, "Data must be divisible with mini_batches"

    step = 0
    errors = np.empty(epochs)
    X = tune.poly_design_matrix(polynomials, data['x_train'])

    beta = init_beta(X.shape[1])
    v = np.zeros_like(beta)

    for epoch in range(1, epochs + 1):
        print('epoch: ', epoch)
        indexes_used = np.zeros(data_size, dtype=bool)
        for mini_batch in range(1, mini_batches + 1):
            k = np.random.choice(np.arange(data_size)[indexes_used == False], size=(int(data_size/mini_batches)))
            indexes_used[k] = True

            y_tilde = X[k] @ beta

            cost_diff = np.dot(X[k].T, 2/(data_size/mini_batches)*(y_tilde - data['y_train'][k]))

            v = momentum * v + learning_rate(step)*cost_diff
            beta = beta - v

            step += 1

        errors[epoch-1] = metrics.MSE(X @ beta, data['y_train'])

    return beta, errors


if __name__ == '__main__':
    import real_terrain

    data = real_terrain.get_data(20)
    print(data['x_train'].shape)

    epochs = 1000
    X_test = tune.poly_design_matrix(6, data['x_train'])

    for i, learning_rate in enumerate(np.logspace(-7, -0.8, 5)):
        beta, errors = sgd(data, 20, epochs, lambda t: learning_rate, 6, 0.5)
        plt.semilogy(errors, label=f"$\\gamma$={learning_rate:.3e}")
        print(f"{learning_rate}:\n Error: {metrics.MSE(data['y_train'], X_test @ beta)}\n beta: {beta}")
    
    model = linear_models.LinearRegression()
    model.fit(X_test, data['y_train'])

    plt.plot([0, epochs], np.repeat(metrics.MSE(model.predict(X_test), data['y_train']), 2), label="Analytical")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("../plots/sgd_different_learning_rates")
    plt.show()



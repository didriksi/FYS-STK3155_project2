import numpy as np
import metrics
import neural_model as NN
import real_terrain

def sdg(model, x, y, epochs=50, mini_batch_size=1, metric=metrics.MSE):

    data_size = x.shape[0]
    indexes = np.arange(data_size)
    errors = np.empty((epochs, 1))

    for epoch in range(epochs):
        print("\r        ", 'epoch: 0| ', f"|{epoch + 1}", end="")
        errors[epoch] = metrics.MSE(model.predict(x), y[:, np.newaxis])  # TODO: Check shapes

        np.random.shuffle(indexes)
        mini_batches = indexes.reshape(-1, mini_batch_size)
        for mini_batch in mini_batches:
            ytilde = model.predict(x[mini_batch])
            model.update_parameters(y[mini_batch][:, np.newaxis])

    print("")

    return model, errors

if __name__ == "__main__":
    model = NN.Network(3)

    model.addLayer(NN.Input(2))
    model.addLayer(NN.Dense(10))
    model.addLayer(NN.Output(1))
    model.compile()

    print(model)

    data = real_terrain.get_data(20)
    x = data['x_train']
    y = data['y_train']
    print("---------")
    print(x.shape)
    errors = sdg(model, x, y, mini_batch_size=20)[1]

    print(model.predict(x).shape)
    print(y.shape)
    print(y[:, np.newaxis].shape)

    import matplotlib.pyplot as plt
    plt.plot(errors)
    plt.show()


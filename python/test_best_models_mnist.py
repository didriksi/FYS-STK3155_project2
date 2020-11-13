import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sgd import sgd
import neural_model
import activations
import learning_rate
import mnist
import metrics
from sklearn.metrics import confusion_matrix


data = mnist.get_data(0.6, 0.2)

epochs = 15000
epochs_without_progress = 500
mini_batch_size = 40

total_steps = epochs * len(data['x_train'])//mini_batch_size

x_train = data['x_train']
y_train = data['y_train']
x_val = data['x_validate']
y_val = data['y_validate']
x_test = data['x_test']
y_test = data['y_test']

learning_rates = [
            learning_rate.Learning_rate(base=base, decay=decay).ramp_up(10).compile(total_steps)
            for base, decay in [(0.005, 0.0001), (0.002, 5e-05)]]

model1 = neural_model.Network(momentum=0.6, learning_rate=learning_rates[0], name="64x64x10_ReLu")
model2 = neural_model.Network(momentum=0.6, learning_rate=learning_rates[1], name="64x32x10_ReLu")

model1.addLayer(neural_model.Input(64))
model1.addLayer(neural_model.Dense(64, activations=activations.relus))
model1.addLayer(neural_model.Output(10, d_func=lambda a, y, _: y - a))
model1.compile()

model2.addLayer(neural_model.Input(64))
model2.addLayer(neural_model.Dense(32, activations=activations.relus))
model2.addLayer(neural_model.Output(10, d_func=lambda a, y, _: y - a))
model2.compile()

errors_1 = sgd(model1, x_train, x_test, y_train, y_test,
               epochs=5000, epochs_without_progress=500,
               mini_batch_size=40, metric=metrics.accuracy)[1]
errors_2 = sgd(model1, x_train, x_test, y_train, y_test,
               epochs=5000, epochs_without_progress=500,
               mini_batch_size=40, metric=metrics.accuracy)[1]

print("Model1: 64x64x10, ReLu activation")
print("    Accuracy: ", errors_1[-1, -1])

print("Model2: 64x32x10, ReLu activation")
print("    Accuracy: ", errors_2[-1, -1])

true_test = np.where(y_test == 1)[1]
predictions = [model1.class_predict(x_test), model2.class_predict(x_test)]
models = [model1, model2]

for i in range(2):
    cf = confusion_matrix(true_test, predictions[i])
    sns.heatmap(cf, annot=True, cbar=False)
    plt.title(f"Confusion matrix - {models[i].name} \n {len(x_test)} samples of validation data")
    plt.savefig(f"../plots/classification_cf_{models[i].name}")
    plt.close()

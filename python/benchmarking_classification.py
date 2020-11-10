import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression

import wisconsin_cancer
import mnist


np.random.seed(10)
cancer_data = wisconsin_cancer.get_data(0.7, 0.2)
mnistData = mnist.get_data(0.7, 0.2)


datasets = [[mnistData, "MNIST"], [cancer_data, "Wisconsin cancer data"]]

for dataset, name in datasets:
    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_test = dataset['x_test']
    y_test = dataset['y_test']

    model = Sequential()
    model.add(Dense(x_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=10, verbose=0)

    pred_train = model.predict(x_train)
    scores = model.evaluate(x_train, y_train, verbose=0)

    print('Model: Neural network, Data: ', name)
    print(f'  Accuracy on training data: {scores[1] * 100:.2f}%')

    pred_test = model.predict(x_test)
    scores2 = model.evaluate(x_test, y_test, verbose=0)
    print(f'  Accuracy on test data: {scores2[1] * 100:.2f}% \n  ')


    # Logistic regression
    y_train = np.where(dataset['y_train'] == 1)[1]
    y_test = np.where(dataset['y_test'] == 1)[1]

    clf = LogisticRegression(random_state=0, max_iter=3000).fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    score2 = clf.score(x_train, y_train)

    print("Model: Logistic regression, Data: ", name)
    print(f"  Accuracy on training data: {score2 * 100:.2f}%")
    print(f"  Accuracy on test data: {score * 100:.2f}% \n ")

    plt.style.use('ggplot')
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('keras model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('../plots/benchmarks/keras_accuracy_'+name)

    plt.close()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('keras model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('../plots/benchmarks/keras_loss_'+name)

    plt.close()

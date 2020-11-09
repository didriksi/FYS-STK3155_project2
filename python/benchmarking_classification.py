from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

import mnist
np.random.seed(10)
mnistData = mnist.get_data(0.8, 0.2)

# Data
x_train = mnistData['x_train']
y_train = mnistData['y_train']
x_test = mnistData['x_test']
y_test = mnistData['y_test']

# Neural network
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=10, verbose=1)

pred_train = model.predict(x_train)
scores = model.evaluate(x_train, y_train, verbose=0)

print('Model: Neural network')
print(f'Accuracy on training data: {scores[1]*100:.2f}%')

pred_test= model.predict(x_test)
scores2 = model.evaluate(x_test, y_test, verbose=0)
print(f'Accuracy on test data: {scores2[1]*100:.2f}%')


#Plot accuracy and loss of neural network
plt.style.use('ggplot')
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('keras model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('../plots/keras_accuracy')

plt.close()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('keras model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('../plots/keras_loss')


# Logistic regression

from sklearn.linear_model import LogisticRegression

#Reformating of data for logistic regression
y_train = np.where(mnistData['y_train'] == 1)[1]
y_test = np.where(mnistData['y_test'] == 1)[1]

clf = LogisticRegression(random_state=0, max_iter=3000).fit(x_train, y_train)
score = clf.score(x_test, y_test)
score2 = clf.score(x_train, y_train)

print("Model: Logistic regression")
print(f"Accuracy on training data: {score2*100:.2f}%")
print(f"Accuracy on test data: { score*100:.2f}%")



from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

import mnist
mnistData = mnist.get_data(0.8, 0.2)

x_train = mnistData['x_train']
y_train = mnistData['y_train']
x_test = mnistData['x_test']
y_test = mnistData['y_test']

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

print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

pred_test= model.predict(x_test)
scores2 = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))


plt.style.use('ggplot')
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('../plots/keras_accuracy')

plt.close()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('../plots/keras_loss')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(25)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)

plt.imshow(x_train[0])
plt.title(y_train[0])

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train/255.0
x_test = x_test/255.0

x_train.shape

y_train[0]

num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

y_train[0]

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
BatchNormalization(axis = -1)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
BatchNormalization(axis = -1)
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
BatchNormalization(axis = -1)
model.add(Conv2D(10, (1, 1)))
BatchNormalization()
model.add(Flatten())
model.add(Activation('softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 5)

result = model.evaluate(x_test, y_test)

print("Accuracy of testset is", result[1])


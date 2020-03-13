from tensorflow.python.keras import layers
import tensorflow as tf

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))


import numpy as np
flatten = layers.Flatten(input_shape=(3,3,1))
data = np.random.random((1, 3, 3, 1))
model = tf.keras.Sequential([flatten])
out = model.forward(data)
print(data)


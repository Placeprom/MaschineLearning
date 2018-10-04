import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(keras.layers.Dense(64, activation='relu'))
# Add another:
model.add(keras.layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(keras.layers.Dense(10, activation='softmax'))
# Configure a model for mean-squared error regression
model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
              loss='mse',
              metrics=['accuracy'])

# Create some data to train the model
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)

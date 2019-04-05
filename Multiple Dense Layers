# Import necessary libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt

# Creat training examples
# Creat raw_x and raw_y
raw_x = list(range(1,21))
print(raw_x)
raw_y = []
for i in raw_x:
  raw_y.append(i**2+3)
print(raw_y)

# Convert raw_x and raw_y into arrays
array_x = np.array(raw_x, dtype=float)
array_y = np.array(raw_y, dtype=float)
#print(array_x, array_y)

# Create the model: Dense network
# Create layers: L zero through L two
l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=4)
l3 = tf.keras.layers.Dense(units=1)

# Assemble the layers into the model
model = tf.keras.Sequential([l0, l1, l2, l3])

# Compile the model with loss and optimizer functions
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# Train the model
history = model.fit(array_x, array_y, epochs=500, verbose=False)
print("Model training completed!")

# Display training statistics
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(history.history['loss'])

# Display layer weights
print(l0.get_weights())
print(l1.get_weights())
print(l2.get_weights())
print(l3.get_weights())

# Use the model to do predictions
print(model.predict([22]))

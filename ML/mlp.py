import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import math
import datetime
import platform

#print('Python version:', platform.python_version())
#print('Tensorflow version:', tf.__version__)
#print('Keras version:', tf.keras.__version__)

mnist_dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()

#print('x_train:', x_train.shape)
#print('y_train:', y_train.shape)
#print('x_test:', x_test.shape)
#print('y_test:', y_test.shape)

#plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()

""" 
numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10,10))
for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()
"""

x_train_normalized = x_train / 255
x_test_normalized = x_test / 255

#plt.imshow(x_train_normalized[0], cmap=plt.cm.binary)
#plt.show()
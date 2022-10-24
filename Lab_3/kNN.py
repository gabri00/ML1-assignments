import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist

# Load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Build a kNN classifier
class kNN:
   def __init__(self, train_X, train_y, test_X, k, test_y=None):
      self.x_train = train_X
      self.y_train = train_y
      self.x_test = test_X
      self.k = k

      # If test_y is not None, then use it as the target
      if test_y is not None:
         self.y_target = test_y

      self.check_data()
   
   def check_data(self):
      # Check that the number of arguments is correct
      # if len() != 6:
      #    raise ValueError("The number of arguments is incorrect!")
      # Check if the columns of the training and test data are the same
      if self.x_train.shape[1] != self.x_test.shape[1]:
         raise ValueError("The number of columns in the training and test data is different!")
      # Check if k is in the correct range
      if self.k < 1 or self.k > self.x_train.shape[0]:
         raise ValueError("k is not in the correct range!")

   def predict(self):
      y_pred = np.zeros(self.x_test.shape[0])

      for i in range(self.x_test.shape[0]):
            # Calculate the distance between the test image and all training images
            dist = np.sum(np.abs(self.x_test[i] - self.x_train), axis=1)
            # Get the k nearest neighbors
            k_nearest_neighbors = np.argsort(dist)[:self.k]
            # Convert to 1D array
            k_nearest_neighbors = k_nearest_neighbors.reshape(-1)
            # Get the most frequent label
            y_pred[i] = np.argmax(np.bincount(self.y_train[k_nearest_neighbors]))

      return y_pred

knn_classifier = kNN(x_train, y_train, x_test, 5, y_test)
print('Predicting...')
prediction = knn_classifier.predict()

print("Accuracy: ", np.sum(prediction == y_test) / y_test.shape[0])
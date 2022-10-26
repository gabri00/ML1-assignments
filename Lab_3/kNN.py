import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist

# Load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print the first 10 elements of the training set
print(x_train[:10])
print(y_train[:10])

# Build a kNN classifier
class kNN:
   def __init__(self, train_X, train_y, test_X, k, test_y=None):
      self.x_train = train_X
      self.y_target = train_y
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
      # Select a random query point
      query_point = np.random.choice(self.x_test.shape[0])
      print(query_point)

      # Calculate the distance between the query point and all the training points
      distances = np.sqrt(np.sum((self.x_train - self.x_test[query_point])**2, axis=1))

      # Sort the distances and get the indices of the k nearest neighbors
      k_indices = np.argsort(distances)[:self.k]

      # Get the labels of the k nearest neighbors
      k_labels = self.y_target[k_indices]

      # Get the most common label
      prediction = np.bincount(k_labels).argmax()

      return prediction

knn_classifier = kNN(x_train[:100], y_train[:100], x_test[:100], 5, y_test[:100])
print('Predicting...')
prediction = knn_classifier.predict()

# Compute the accuracy on the test set on each digit vs all other digits
# for i in range(10):
#    y_test_i = (y_test == i).astype(int)
#    y_pred_i = (prediction == i).astype(int)
#    print("Accuracy for digit {}: {}".format(i, np.sum(y_pred_i == y_test_i) / y_test_i.shape[0]))
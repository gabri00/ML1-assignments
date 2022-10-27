import numpy as np
from collections import Counter
from tensorflow.keras.datasets import mnist

# Load mnist dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Create functions for different distance metrics
def euclidean_distance(x1, x2):
   return np.sqrt(np.sum((x1 - x2)**2))

def manhattan_distance(x1, x2):
   return np.sum(np.abs(x1 - x2))

# Function to check the data
def check_data(train_X, test_X, k):
   # Check if the columns of the training and test data are the same
   if train_X.shape[1] != test_X.shape[1]:
      raise ValueError("The number of columns in the training and test data is different!")
   # Check if k is in the correct range
   if k < 1 or k > train_X.shape[0]:
      raise ValueError("k is not in the correct range!")

# Create function for kNN
def kNN(train_X, train_y, test_X, k, test_y=None):
   check_data(train_X, test_X, k)

   # If y_test is not None, use it as the target   
   test_y = test_y if not None else train_y[:test_X.shape[0]]

   # Calculate the distance between the test points and the training points
   distances = np.zeros((test_X.shape[0], train_X.shape[0]))
   for i in range(test_X.shape[0]):
      for j in range(train_X.shape[0]):
         distances[i, j] = euclidean_distance(test_X[i], train_X[j])
   
   # Find the k nearest neighbors
   k_neighbors = np.zeros((test_X.shape[0], k))
   for i in range(test_X.shape[0]):
      k_neighbors[i] = np.argsort(distances[i])[:k]
   
   # Find the most common class in the k nearest neighbors
   y_pred = np.zeros(test_X.shape[0])
   for i in range(test_X.shape[0]):
      y_pred[i] = Counter(train_y[k_neighbors[i].astype(int)]).most_common(1)[0][0]

   # Compute and return the error rate
   return np.sum(y_pred != test_y) / test_y.shape[0]


# Select the first 100 images from the training set
train_X = train_X[:1000]
train_y = train_y[:1000]

# Select the first 10 images from the test set
test_X = test_X[:100]
test_y = test_y[:100]

# Predict the labels of the test set for k = 1,2,3,4,5,10,15,20,30,40,50
for k in [1,2,3,4,5,10,15,20,30,40,50]:
   error_rate = kNN(train_X, train_y, test_X, k, test_y)
   print("k = {}, error rate = {}%".format(k, error_rate*100))

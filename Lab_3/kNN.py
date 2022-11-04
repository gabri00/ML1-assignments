import numpy as np
import matplotlib.pyplot as plt
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

   # Compute and return the error rate if test_y is given
   if test_y is not None:
      return y_pred, np.sum(y_pred != test_y) / test_X.shape[0]
   else:
      return y_pred


# Select a random subset of the training data of size 1000
idx = np.random.permutation(train_X.shape[0])[:100]
train_X = train_X[idx]
train_y = train_y[idx]

# Predict each digit from the test set for every k and calculate the accuracy
accuracies = {}
k = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]

for i in range(10):
   # Print the progress
   test_X_i = test_X[test_y == i][:50]
   test_y_i = test_y[test_y == i][:50]
   accuracies[i] = []
   for k_i in k:
      accuracies[i].append((1 - kNN(train_X, train_y, test_X_i, k_i, test_y_i)[1])*100)

# Plot the error rate for each digit and each k
for i in range(10):
   plt.plot(k, accuracies[i], label=f'Digit {i}')
plt.xlabel('k')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

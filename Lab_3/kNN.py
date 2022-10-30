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


# Select 10000 samples from the dataset
train_X = train_X[:5000]
train_y = train_y[:5000]

# Predict one digit from the test set
error_rates = []
for i in range(10):
   test_X_i = test_X[test_y == i][:50]
   test_y_i = test_y[test_y == i][:50]
   error_rates.append(kNN(train_X, train_y, test_X_i, 3, test_y_i))
   print(f'Digit #{i} predicted with error {error_rates[i]*100}%')

# Plot the error rates
plt.plot(range(10), error_rates)
plt.xlabel("Digit")
plt.ylabel("Error rate")
plt.show()

# Predict the labels of the test set for k = 1,2,3,4,5,10,15,20,30,40,50
# k = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
# error_rate = [kNN(train_X, train_y, test_X, k_i, test_y) for k_i in k]

# for k_i, e in zip(k, error_rate):
#    print("k = {}, error rate = {}%".format(k_i, e*100))

# Make a graph of the error rate for different values of k
# plt.plot(k, error_rate)
# plt.xlabel("k")
# plt.ylabel("Error rate")
# plt.show()

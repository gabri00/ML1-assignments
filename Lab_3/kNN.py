import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

### Task 1 ###

# Load mnist dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()


### Task 2 ###

# Function for Euclidean distance
def euclidean_distance(x1, x2):
   # return np.sqrt(np.sum((x1 - x2)**2))
   return np.linalg.norm(x1 - x2)

# Function to check the data
def check_data(train_X, test_X, k):
   # Check if the number of columns in the training and test sets are the same
   if train_X.shape[1] != test_X.shape[1]:
      raise ValueError("The number of columns in the training and test sets is different!")
   # Check if k is in the correct range
   if k < 1 or k > train_X.shape[0]:
      raise ValueError("k is not in the correct range!")

# Function for k-Nearst Neighbors algorithm
def kNN(train_X, train_y, test_X, k, test_y=None):
   check_data(train_X, test_X, k)

   # Calculate the distances between the test images and the training images
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
      y_pred[i] = np.argmax(np.bincount(train_y[k_neighbors[i].astype(int)]))

   # Compute and return the error rate and the predicted labels if test_y is given
   # otherwise return only the predicted labels
   if test_y is not None:
      return y_pred, np.sum(y_pred != test_y) / test_X.shape[0]
   else:
      return y_pred.astype(int)


# Select a random subset (5%) of the training data to reduce the computation time
idx = np.random.permutation(train_X.shape[0])[:int(round(test_X.shape[0] * 0.05))]
train_X = train_X[idx]
train_y = train_y[idx]

# Classify 5% of the test data with k=5 (test_y is not given)
idx = np.random.permutation(test_X.shape[0])[:int(round(test_X.shape[0] * 0.05))]

# prediction = kNN(train_X, train_y, test_X[idx], 5)
# print("Error rate: ", np.sum(prediction != test_y[idx]) / len(prediction))

# Classify 5% of the test data with k=5 (test_y is given)
# _, error_rate = kNN(train_X, train_y, test_X[idx], 5, test_y[idx])
# print("Error rate: ", error_rate)


### Task 3 & Addendum ###

# Create a function to calculate the confusion matrix
def confusion_matrix(y_true, y_pred):
   # Create a confusion matrix
   cm = np.zeros((2, 2))

   for i in range(len(y_true)):
      cm[1][int(y_true[i] == y_pred[i])] += 1

   return cm

classes = np.unique(test_y)
k = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
accuracies = []
# 2x2 confusion matrix for each class and each k
confusion_matrices = np.zeros((len(classes), len(k), 2, 2))

# Predict each digit from the test set for every k and calculate the accuracy
for i in range(len(classes)):
   test_X_i = test_X[test_y == i]
   test_y_i = test_y[test_y == i]
   # Select a random 10% of test data to reduce the computation time
   idx = np.random.permutation(test_X_i.shape[0])[:int(round(test_X_i.shape[0] * 0.1))]
   test_X_i = test_X_i[idx]
   test_y_i = test_y_i[idx]

   accuracies.append([])
   for j in range(len(k)):
      y_pred, err_rate = kNN(train_X, train_y, test_X_i, k[j], test_y_i)
      accuracies[i].append((1 - err_rate))
      confusion_matrices[i][j] = confusion_matrix(test_y_i, y_pred)

   # Plot the error rate for each digit and each k
   plt.plot(k, accuracies[i], label=f'Digit {i}')
   # Print the progress
   print("Progress: ", i + 1, "/ 10")

plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Calculate sensitivity and F1 score for each digit and each k
# Note: specificity cannot be calculated and the precision is always 1
sensitivity = np.zeros((len(classes), len(k)))
f1_score = np.zeros((len(classes), len(k)))
for i in range(len(classes)):
   for j in range(len(k)):
      sensitivity[i][j] = confusion_matrices[i][j][1][1] / (confusion_matrices[i][j][1][1] + confusion_matrices[i][j][1][0])
      f1_score[i][j] = 2 * sensitivity[i][j] / (sensitivity[i][j] + 1)

# Plot the sensitivity and F1 score for each digit and each k
for i in range(len(classes)):
   plt.plot(k, sensitivity[i], label=f'Digit {i}')
plt.xlabel('k')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()

for i in range(len(classes)):
   plt.plot(k, f1_score[i], label=f'Digit {i}')
plt.xlabel('k')
plt.ylabel('F1 score')
plt.legend()
plt.show()

# Calculate the average sensitivity and F1 score for each digit
avg_sensitivity = np.zeros(len(classes))
avg_f1_score = np.zeros(len(classes))
for i in range(len(classes)):
   avg_sensitivity[i] = np.mean(sensitivity[i])
   avg_f1_score[i] = np.mean(f1_score[i])

# Plot the average sensitivity and F1 score for each digit in a subplot
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Average sensitivity and F1 score for each digit')
ax1.bar(classes, avg_sensitivity)
ax1.set_xlabel('Digit')
ax1.set_ylabel('Average sensitivity')
ax2.bar(classes, avg_f1_score)
ax2.set_xlabel('Digit')
ax2.set_ylabel('Average F1 score')
plt.show()

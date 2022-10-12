# Linear regression - Lab 2
# 2.1. Linear regression one dimensional without intercept on df1
# 2.2. Compare (plot) the solution obtained on different random subsets (10%)
# 2.3. Linear regression one dimensional with intercept on df2
# 2.4. Multidimensional linear regression on df2 (mpg as target)

from random import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

########## Task 1 ##########

# Set dataset path
PATH_DF_1 = 'turkish-se-SP500vsMSCI.csv'
PATH_DF_2 = 'mtcarsdata-4features.csv'

# Read the data from the csv file
df1 = pd.read_csv(PATH_DF_1, header=None)
df2 = pd.read_csv(PATH_DF_2)

# Set df names
df1.name = 'SP500vsMSCI'
df2.name = 'mtcarsdata'

# Remove spaces from the column names
df2.columns = df2.columns.str.strip()

########## Task 2 ##########

######### Helper functions #########

# Function for linear regression one dimensional WITHOUT intercept
def linear_regression_one_dim_no_intercept(df, x_col, y_col):
      # Calculate numerator and denominator of the slope
      numerator = 0
      denominator = 0
      for x, y in zip(df[x_col], df[y_col]):
         numerator += x * y
         denominator += x**2
      # Calculate the slope
      slope = numerator / denominator
      # Return the linear regression function
      return lambda x: slope * x

# Function for linear regression one dimensional WITH intercept
def linear_regression_one_dim(df, x_col, y_col):
      # Calculate mean of x and y
      x_mean = df[x_col].mean()
      y_mean = df[y_col].mean()
      # Calculate numerator and denominator of the slope
      numerator = 0
      denominator = 0
      for x, y in zip(df[x_col], df[y_col]):
         numerator += (x - x_mean) * (y - y_mean)
         denominator += (x - x_mean)**2
      # Calculate the slope
      if denominator == 0:
         slope = 0
      else:
         slope = numerator / denominator
      # Calculate the intercept
      intercept = y_mean - (slope * x_mean)
      # Return the linear regression function
      return lambda x: slope * x + intercept

# Function for multidimensional linear regression
def linear_regression_multidim(df, x_cols, target_col):
      y = df[target_col].values
      X = df[x_cols].values
      # If the matrix is singular and cannot be inverted return a linear regression function that returns 0
      if X.shape[0] == 1:
         return lambda x: 0
      # Calculate the Moore-Penrose pseudo-inverse
      # X_pinv = np.linalg.pinv(X)
      X_pinv = np.linalg.inv(X.T @ X) @ X.T
      # Calculate the weights
      weights = X_pinv @ y
      # Return the linear regression function
      return lambda x: x @ weights

# Function to plot the data and the linear regression line (one dimensional)
def plot_linear_regression(df, x_col, y_col, lin_reg, xlabel, ylabel, legend=True):
      # Plot the data
      plt.scatter(df[x_col], df[y_col], marker='x')
      # Plot the linear regression line
      x = np.linspace(df[x_col].min(), df[x_col].max(), 100)
      plt.plot(x, lin_reg(x), color='red')

      if legend:
         plt.legend(['Data', 'Linear regression'])
      plt.title(f'One dimensional linear regression on {x_col}, {y_col}')
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.show()

# Function to make a n subplots of the linear regression line
def plot_linear_regression_subplots(df_subsets, x_col, y_col, lin_reg, n, xlabel, ylabel, legend=False):
   fig, axs = plt.subplots(n // 2, n // 2)
   fig.suptitle(f'Linear Regression on subsets')

   for i in range(n):
      # Get the linear regression function
      linear_regression = lin_reg(df_subsets[i], x_col, y_col)
      # Plot the data
      axs[i // 2, i % 2].scatter(df_subsets[i][x_col], df_subsets[i][y_col], marker='x')
      # Plot the linear regression line
      x = np.linspace(df_subsets[i][x_col].min(), df_subsets[i][x_col].max(), 100)
      axs[i // 2, i % 2].plot(x, linear_regression(x), color='red')

      if legend:
         axs[i // 2, i % 2].legend(['Linear regression', 'Data'])
      axs[i // 2, i % 2].set_xlabel(xlabel)
      axs[i // 2, i % 2].set_ylabel(ylabel)

   plt.show()

# Function to split the data into training and test sets
def split_data(df, percentage):
      # Get the random subset
      df_subset = np.random.permutation(df.shape[0])[:int(df.shape[0] * percentage)]
      # Split the data
      train_set = df.iloc[df_subset]
      test_set = df.iloc[~df_subset]

      return train_set, test_set

######### End of helper functions #########


######### Configurations #########

# df1
x_col_df1, y_col_df1 = 0, 1
df1_labels = ['SP500', 'MSCI']

# df2
x_col_df2, y_col_df2 = 'weight', 'mpg'
df2_labels = ['Car weight', 'Fuel efficiency (mpg)']

# df2 - Multidimensional
# Drop the first column (name) and the target column (mpg)
x_cols_df2 = df2.drop([df2.columns[0], y_col_df2], axis=1).columns

######### End of configurations #########


# 2.1. Linear regression one dimensional without intercept on df1
lin_reg = linear_regression_one_dim_no_intercept(df1, x_col_df1, y_col_df1)
# Plot the linear regression line for df1
plot_linear_regression(df1, x_col_df1, y_col_df1, lin_reg, df1_labels[0], df1_labels[1])


# 2.2. Compare (plot) the solution obtained on different random subsets (10%)
n = 4
# Get n random subsets of df1
df1_subsets = [split_data(df1, 0.1) for _ in range(n)]
df1_subsets = [subset[0] for subset in df1_subsets]
# Make n subplots with the subsets
plot_linear_regression_subplots(
   df1_subsets,
   x_col_df1,
   y_col_df1,
   linear_regression_one_dim_no_intercept,
   n,
   df1_labels[0],
   df1_labels[1])


# 2.3. Linear regression one dimensional with intercept on df2
lin_reg = linear_regression_one_dim(df2, x_col_df2, y_col_df2)
# Plot the linear regression line for df2
plot_linear_regression(df2, x_col_df2, y_col_df2, lin_reg, df2_labels[0], df2_labels[1])


# 2.4. Multidimensional linear regression on df2 (mpg as target)
lin_reg = linear_regression_multidim(df2, x_cols_df2, y_col_df2)
# Predict and print the mpg for df2
mpg_predicted = df2[x_cols_df2].apply(lin_reg, axis=1)
print(mpg_predicted)


######### Task 3 #########

# Repeat 2.1, 2.3, 2.4 with 5% of the data
# get 5% of the data for df1 and df2
train_set_df1, test_set_df1 = split_data(df1, 0.05)
train_set_df2, test_set_df2 = split_data(df2, 0.05)

# 3.1. Linear regression one dimensional without intercept on df1
lin_reg_df1 = linear_regression_one_dim_no_intercept(train_set_df1, x_col_df1, y_col_df1)
# Plot the linear regression line for df1
plot_linear_regression(train_set_df1, x_col_df1, y_col_df1, lin_reg_df1, df1_labels[0], df1_labels[1])

# 3.2. Linear regression one dimensional with intercept on df2
lin_reg_df2 = linear_regression_one_dim(train_set_df2, x_col_df2, y_col_df2)
# Plot the linear regression line for df2
plot_linear_regression(train_set_df2, x_col_df2, y_col_df2, lin_reg_df2, df2_labels[0], df2_labels[1])

# 3.3. Multidimensional linear regression on df2 (mpg as target)
lin_reg_df2_multidim = linear_regression_multidim(train_set_df2, x_cols_df2, y_col_df2)
# Predict and print the mpg for df2
mpg_predicted = train_set_df2[x_cols_df2].apply(lin_reg_df2_multidim, axis=1)
print(mpg_predicted)

# Function to compute the mean squared error
def mean_squared_error(df, x_col, y_col, lin_reg):
   # Predict the target
   y_pred = df[x_col].apply(lin_reg)
   # Compute the mean squared error
   return np.mean((df[y_col] - y_pred)**2)

# 3.4. Compute the mean squared error for the train set of df1 and df2
mse_train_set_df1 = mean_squared_error(train_set_df1, x_col_df1, y_col_df1, lin_reg_df1)
mse_train_set_df2 = mean_squared_error(train_set_df2, x_col_df2, y_col_df2, lin_reg_df2)

# 3.5. Compute the mean squared error for the test set of df1 and df2
mse_test_set_df1 = mean_squared_error(test_set_df1, x_col_df1, y_col_df1, lin_reg_df1)
mse_test_set_df2 = mean_squared_error(test_set_df2, x_col_df2, y_col_df2, lin_reg_df2)

# Print the mean squared error for the train and test sets of df1 and df2
print(f'MSE for the train set of df1: {mse_train_set_df1}')
print(f'MSE for the test set of df1: {mse_test_set_df1}')
print(f'MSE for the train set of df2: {mse_train_set_df2}')
print(f'MSE for the test set of df2: {mse_test_set_df2}')

# 3.6. repeat with random n% of the data
n = 10
percentages = [np.random.permutation(100) / 100 for _ in range(n)]
df1_sets = []
df2_sets = []

for p in percentages:
   df1_sets.append(split_data(df1, p))
   df2_sets.append(split_data(df2, p))

df1_sets = [subset[0] for subset in df1_sets]
df2_sets = [subset[0] for subset in df2_sets]

# Function to plot multiple linear regression lines
def plot_multiple_linear_regressions(df_subsets, x_col, y_col, lin_reg, n, xlabel, ylabel, legend=True):
   # Plot the data
   plt.scatter(df1[x_col], df1[y_col], marker='x')
   
   for i in range(n):
      # Plot the linear regression line
      x = np.linspace(df_subsets[i][x_col].min(), df_subsets[i][x_col].max(), 100)
      plt.plot(x, lin_reg(x), color='red')

   if legend:
      plt.legend(['Data', 'Linear regression'])
   plt.title(f'One dimensional linear regression on {x_col}, {y_col}')
   plt.xlabel(xlabel)
   plt.ylabel(ylabel)
   plt.show()

# Plot all linear regression lines for df1
plot_multiple_linear_regressions(df1_sets, x_col_df1, y_col_df1, linear_regression_one_dim_no_intercept, n, df1_labels[0], df1_labels[1])

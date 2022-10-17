# Linear regression - Lab 2
# 2.1. Linear regression one dimensional without intercept on df1
# 2.2. Compare (plot) the solution obtained on different random subsets (10%)
# 2.3. Linear regression one dimensional with intercept on df2
# 2.4. Multidimensional linear regression on df2 (mpg as target)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

########## Task 1 ##########

# Set datasets path
PATH_DF1 = 'turkish-se-SP500vsMSCI.csv'
PATH_DF2 = 'mtcarsdata-4features.csv'

# Read the data from the csv files
df1 = pd.read_csv(PATH_DF1, header=None)
df2 = pd.read_csv(PATH_DF2)

# Set df names
df1.name = 'Stock Exchange (SP500vsMSCI)'
df2.name = 'Motor Trend Car Road Tests'

# Remove useless spaces from the column names
df2.columns = df2.columns.str.strip()

########## Task 2 ##########

######### Helper functions #########

# Function for linear regression one dimensional WITHOUT intercept
def linear_regression_one_dim_no_intercept(df, x_col, y_col):
      # Calculate numerator and denominator of the slope
      num = 0
      den = 0
      for x, y in zip(df[x_col], df[y_col]):
         num += x * y
         den += x**2
      # Calculate the slope
      slope = num / den
      # Return the linear regression function
      return lambda x: slope * x

# Function for linear regression one dimensional WITH intercept
def linear_regression_one_dim(df, x_col, y_col):
      # Calculate mean of x and y
      x_mean = df[x_col].mean()
      y_mean = df[y_col].mean()
      # Calculate numerator and denominator of the slope
      num = 0
      den = 0
      for x, y in zip(df[x_col], df[y_col]):
         num += (x - x_mean) * (y - y_mean)
         den += (x - x_mean)**2
      # Calculate the slope
      if den == 0:
         slope = 0
      else:
         slope = num / den
      # Calculate the intercept
      intercept = y_mean - (slope * x_mean)
      # Return the linear regression function
      return lambda x: slope * x + intercept

# Function for multidimensional linear regression
def linear_regression_multidim(df, x_cols, target_col):
      y = df[target_col].values
      X = df[x_cols].values
      # If the matrix is singular and cannot be inverted return a linear regression function that returns 0
      try:
         # Calculate the Moore-Penrose pseudo-inverse
         # X_pinv = np.linalg.pinv(X)
         X_pinv = np.linalg.inv(X.T @ X) @ X.T
         # Calculate the weights
         weights = X_pinv @ y
         # Return the linear regression function
         return lambda x: x @ weights
      except np.linalg.LinAlgError:
         return lambda x: 0

# Function to plot the data and the linear regression line
def plot_linear_regression(df, x_col, y_col, lin_reg, xlabel, ylabel, title, legend=True):
      # Plot the data
      plt.scatter(df[x_col], df[y_col], marker=DATA_MARKER)
      # Plot the linear regression line
      x = np.linspace(df[x_col].min(), df[x_col].max(), 100)
      plt.plot(x, lin_reg(x), color=LR_COLOR)

      if legend:
         plt.legend(['Data', 'Linear regression'])
      plt.title(title)
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      plt.show()

# Function to make a n subplots of the linear regression line
def plot_linear_regression_subplots(df_subsets, x_col, y_col, lin_reg, n, xlabel, ylabel, title, legend=False):
   fig, axs = plt.subplots(n // 2, n // 2)
   fig.suptitle(title)

   for i in range(n):
      curr_axs = axs[i // 2, i % 2]
      curr_subset = df_subsets[i]
      # Get the linear regression function
      linear_regression = lin_reg(curr_subset, x_col, y_col)
      # Plot the data
      curr_axs.scatter(curr_subset[x_col], curr_subset[y_col], marker=DATA_MARKER)
      # Plot the linear regression line
      x = np.linspace(curr_subset[x_col].min(), curr_subset[x_col].max(), 100)
      curr_axs.plot(x, linear_regression(x), color=LR_COLOR)

      if legend:
         curr_axs.legend(['Linear regression', 'Data'])
      curr_axs.set_xlabel(xlabel)
      curr_axs.set_ylabel(ylabel)

   plt.show()

# Function to plot multiple linear regression lines
def plot_multiple_linear_regressions(df_subsets, x_col, y_col, lin_reg, n, xlabel, ylabel, title, legend=False):
   c = ['red', 'black', 'brown', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple']
   for i in range(n):
      curr_subplot = df_subsets[i]
      linear_regression = lin_reg(curr_subplot, x_col, y_col)
      # Plot the data
      plt.scatter(curr_subplot[x_col], curr_subplot[y_col], marker=DATA_MARKER, color=c[i])
      # Plot the linear regression line
      x = np.linspace(curr_subplot[x_col].min(), curr_subplot[x_col].max(), 100)
      plt.plot(x, linear_regression(x), color=c[i])

   if legend:
      plt.legend(['Data', 'Linear regression'])
   plt.title(title)
   plt.xlabel(xlabel)
   plt.ylabel(ylabel)
   plt.show()

# Function to split the data into training and test sets
def split_data(df, percentage):
      # Get the random subset
      subset = np.random.permutation(df.shape[0])[:int(round(df.shape[0] * percentage))]
      # Split the data
      train_set = df.iloc[subset]
      test_set = df.iloc[~subset]

      return train_set, test_set

# Function to compute the mean squared error
def mean_squared_error(df, x_col, y_col, lin_reg):
   # Predict the target
   y_pred = df[x_col].apply(lin_reg)
   # Compute the mean squared error
   return np.mean((df[y_col] - y_pred)**2)

######### End of helper functions #########


######### Configurations #########

# df1
x_col_df1, y_col_df1 = 0, 1
df1_labels = ['SP500 return index', 'MSCI Europe index']

# df2
x_col_df2, y_col_df2 = 'weight', 'mpg'
df2_labels = ['Car weight (lbs/1000)', 'Fuel efficiency (mpg)']

# df2 - Multidimensional
# Drop the first column (name) and the target column (mpg)
x_cols_df2 = df2.drop([df2.columns[0], y_col_df2], axis=1).columns

# Plot configurations
DATA_MARKER = '.'
LR_COLOR = 'red'

######### End of configurations #########


# 2.1. Linear regression one dimensional without intercept on df1
lin_reg = linear_regression_one_dim_no_intercept(df1, x_col_df1, y_col_df1)
# Plot the linear regression line for df1
plot_linear_regression(df1, x_col_df1, y_col_df1, lin_reg, df1_labels[0], df1_labels[1], 'Linear regression without intercept on df1')


# 2.2. Compare (plot) the solution obtained on different random subsets (10%)
n = 4
# Get n random subsets of df1
df1_subsets = [split_data(df1, 0.1) for _ in range(n)]
# Get only the training sets
df1_subsets = [subset[0] for subset in df1_subsets]
# Make n subplots with the subsets
plot_linear_regression_subplots(
   df1_subsets,
   x_col_df1,
   y_col_df1,
   linear_regression_one_dim_no_intercept,
   n,
   df1_labels[0],
   df1_labels[1],
   'Solutions obtained on different random subsets (10%) of df1'
)


# 2.3. Linear regression one dimensional with intercept on df2
lin_reg = linear_regression_one_dim(df2, x_col_df2, y_col_df2)
# Plot the linear regression line for df2
plot_linear_regression(df2, x_col_df2, y_col_df2, lin_reg, df2_labels[0], df2_labels[1], 'Linear regression with intercept on df2')


# 2.4. Multidimensional linear regression on df2 (mpg as target)
lin_reg = linear_regression_multidim(df2, x_cols_df2, y_col_df2)
# Predict and print the mpg for df2
mpg_predicted = df2[x_cols_df2].apply(lin_reg, axis=1)
print(mpg_predicted)


######### Task 3 #########

# Repeat 2.1, 2.3, 2.4 with 5% of the data
# get 5% of the data for df1 and df2
p = 0.05
train_set_df1, test_set_df1 = split_data(df1, p)
train_set_df2, test_set_df2 = split_data(df2, p)

# 3.1. Linear regression one dimensional without intercept on df1 with 5% of the data
lin_reg_df1 = linear_regression_one_dim_no_intercept(train_set_df1, x_col_df1, y_col_df1)
# Plot the linear regression line for df1
plot_linear_regression(train_set_df1, x_col_df1, y_col_df1, lin_reg_df1, df1_labels[0], df1_labels[1], 'Linear regression without intercept on df1 (5% of the data)')

# 3.2. Linear regression one dimensional with intercept on df2 with 5% of the data
lin_reg_df2 = linear_regression_one_dim(train_set_df2, x_col_df2, y_col_df2)
# Plot the linear regression line for df2
plot_linear_regression(train_set_df2, x_col_df2, y_col_df2, lin_reg_df2, df2_labels[0], df2_labels[1], 'Linear regression with intercept on df2 (5% of the data)')

# 3.3. Multidimensional linear regression on df2 (mpg as target) with 5% of the data
lin_reg_df2_multidim = linear_regression_multidim(train_set_df2, x_cols_df2, y_col_df2)
# Predict and print the mpg for df2
mpg_predicted = train_set_df2[x_cols_df2].apply(lin_reg_df2_multidim, axis=1)
print(mpg_predicted)

# 3.4. Compute the mean squared error for the train set of df1 and df2
mse_train_set_df1 = mean_squared_error(train_set_df1, x_col_df1, y_col_df1, lin_reg_df1)
mse_train_set_df2 = mean_squared_error(train_set_df2, x_col_df2, y_col_df2, lin_reg_df2)

# 3.5. Compute the mean squared error for the test set of df1 and df2
mse_test_set_df1 = mean_squared_error(
   test_set_df1,
   x_col_df1,
   y_col_df1,
   linear_regression_one_dim_no_intercept(test_set_df1, x_col_df1, y_col_df1)
)
mse_test_set_df2 = mean_squared_error(
   test_set_df2,
   x_col_df2,
   y_col_df2,
   linear_regression_one_dim(test_set_df2, x_col_df2, y_col_df2)
)

# Print the mean squared error for the train and test sets of df1 and df2
print(f'MSE for the train set of df1: {mse_train_set_df1}')
print(f'MSE for the test set of df1: {mse_test_set_df1}')
print(f'MSE for the train set of df2: {mse_train_set_df2}')
print(f'MSE for the test set of df2: {mse_test_set_df2}')

# 3.6. repeat with random n% of the data
n = 10
percentages = np.random.permutation(100)[:n] / 100
df1_sets = []
df2_sets = []

for p in percentages:
   df1_sets.append(split_data(df1, p))
   df2_sets.append(split_data(df2, p))

df1_train_sets = [subset[0] for subset in df1_sets]
df2_train_sets = [subset[0] for subset in df2_sets]
df1_test_sets = [subset[1] for subset in df1_sets]
df2_test_sets = [subset[1] for subset in df2_sets]

# Plot all linear regression lines for df1 train sets
plot_multiple_linear_regressions(
   df1_train_sets,
   x_col_df1,
   y_col_df1,
   linear_regression_one_dim_no_intercept,
   n,
   df1_labels[0],
   df1_labels[1],
   f'Linear regression without intercept on {n} different train subsets (random %) of df1'
)

# Plot all linear regression lines for df1 test sets
plot_multiple_linear_regressions(
   df1_test_sets,
   x_col_df1,
   y_col_df1,
   linear_regression_one_dim_no_intercept,
   n,
   df1_labels[0],
   df1_labels[1],
   f'Linear regression without intercept on {n} different test subsets (random %) of df1'
)

# Plot all linear regression lines for df2 train sets
plot_multiple_linear_regressions(
   df2_train_sets,
   x_col_df2,
   y_col_df2,
   linear_regression_one_dim,
   n,
   df2_labels[0],
   df2_labels[1],
   f'Linear regression with intercept on {n} different train subsets (random %) of df2'
)

# Plot all linear regression lines for df2 test sets
plot_multiple_linear_regressions(
   df2_test_sets,
   x_col_df2,
   y_col_df2,
   linear_regression_one_dim_no_intercept,
   n,
   df2_labels[0],
   df2_labels[1],
   f'Linear regression with intercept on {n} different test subsets (random %) of df2'
)

# In 2.4 do we have to print the points of y ?
# Clarifications on task 3.6
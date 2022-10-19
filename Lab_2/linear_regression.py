# Linear regression - Lab 2

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
      if den == 0:
         slope = 0
      else:
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
      # If the matrix is singular return a linear regression function that returns 0
      try:
         # Calculate the Moore-Penrose pseudo-inverse
         X_pinv = np.linalg.inv(X.T @ X) @ X.T
         # Calculate the weights
         w = X_pinv @ y
         # Return the linear regression function
         return lambda x: x @ w
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
def plot_linear_regression_subplots(df_subsets, x_col, y_col, lin_reg, n, xlabel, ylabel, title):
   fig, axs = plt.subplots(n // 2, n // 2)
   fig.suptitle(title)

   for i, ax in enumerate(axs.flat):
      curr_subset = df_subsets[i]
      # Get the linear regression function
      linear_regression = lin_reg(curr_subset, x_col, y_col)
      # Plot the data
      ax.scatter(curr_subset[x_col], curr_subset[y_col], marker=DATA_MARKER)
      # Plot the linear regression line
      x = np.linspace(curr_subset[x_col].min(), curr_subset[x_col].max(), 100)
      ax.plot(x, linear_regression(x), color=LR_COLOR)

      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)

   plt.show()

# Function to plot multiple linear regression lines
def plot_multiple_linear_regressions(df_subsets, x_col, y_col, lin_reg, xlabel, ylabel, title):
   c = ['red', 'green', 'brown', 'blue', 'black', 'cyan', 'magenta', 'yellow', 'orange', 'purple']
   lin_regs = []

   for i in range(len(df_subsets)):
      curr_subplot = df_subsets[i]
      linear_regression = lin_reg(curr_subplot, x_col, y_col)
      # Plot the data
      plt.scatter(curr_subplot[x_col], curr_subplot[y_col], marker=DATA_MARKER, color=c[i])
      # Plot the linear regression line
      x = np.linspace(curr_subplot[x_col].min(), curr_subplot[x_col].max(), 100)
      plt.plot(x, linear_regression(x), color=c[i])
      lin_regs.append(linear_regression)

   plt.title(title)
   plt.xlabel(xlabel)
   plt.ylabel(ylabel)
   plt.show()

   return lin_regs

# Function to plot mean squared errors in histograms
def mse_histograms(data, title, subtitles, x_label, y_label):
   fig, axs = plt.subplots(len(data) // (len(data) // 2), len(data) // 2)
   fig.suptitle(title)

   for i, ax in enumerate(axs.flat):
      ax.hist(data[i], bins=10, edgecolor='black')
      ax.set_title(subtitles[i])
      ax.set_xlabel(x_label)
      ax.set_ylabel(y_label)

   plt.tight_layout()
   plt.show()

# Function to split the data into training and test sets
def split_data(df, percentage):
      # Get the random subset
      subset = np.random.permutation(df.shape[0])[:int(round(df.shape[0] * percentage))]
      # Split the data
      train_set = df.iloc[subset]
      test_set = df.drop(train_set.index)

      return train_set, test_set

# Function to compute the mean squared error
def mean_squared_error(df, x_col, y_col, lin_reg):
   # Predict the target
   y_pred = df[x_col].apply(lin_reg)
   # Compute the mean squared error
   return np.mean((df[y_col] - y_pred)**2)

# Function to compute the mean squared error (multidimensional)
def mean_squared_error_multidim(df, x_cols, y_col, lin_reg):
   # Predict the target
   y_pred = df[x_cols].apply(lin_reg, axis=1)
   # Compute the mean squared error
   return 0.5 * np.linalg.norm(df[y_col] - y_pred)**2

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

# Lambda to convert percentage to string
percent_to_str = lambda p: str(int(round(p * 100))) + '%'

######### End of configurations #########


# 2.1. Linear regression one dimensional without intercept on df1
lin_reg = linear_regression_one_dim_no_intercept(df1, x_col_df1, y_col_df1)
# Plot the linear regression line for df1
plot_linear_regression(df1, x_col_df1, y_col_df1, lin_reg, df1_labels[0], df1_labels[1], 'Linear regression without intercept on df1')


# 2.2. Compare (plot) the solution obtained on different random subsets (10%)
n = 4
p = 0.1
# Get n random subsets of df1
df1_subsets = [split_data(df1, p) for _ in range(n)]
# Get only the training sets
df1_subsets = [subset[0] for subset in df1_subsets]

# Make a plot with all the subsets
plot_multiple_linear_regressions(
   df1_subsets,
   x_col_df1,
   y_col_df1,
   linear_regression_one_dim_no_intercept,
   df1_labels[0],
   df1_labels[1],
   f'{len(df1_subsets)} random subsets {percent_to_str(p)} of df1'
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

# 3.1. Linear regression one dimensional without intercept on df1 with with train (5%) and test (95%)
lin_reg_train_df1, lin_reg_test_df1 = plot_multiple_linear_regressions(
   [train_set_df1, test_set_df1],
   x_col_df1,
   y_col_df1,
   linear_regression_one_dim_no_intercept,
   df1_labels[0],
   df1_labels[1],
   f'Linear regression without intercept on train {percent_to_str(p)} and test {percent_to_str(1-p)} of df1'
)

# 3.2. Linear regression one dimensional with intercept on df2 with train (5%) and test (95%)
lin_reg_train_df2, lin_reg_test_df2 = plot_multiple_linear_regressions(
   [train_set_df2, test_set_df2],
   x_col_df2,
   y_col_df2,
   linear_regression_one_dim,
   df2_labels[0],
   df2_labels[1],
   f'Linear regression with intercept on train {percent_to_str(p)} and test {percent_to_str(1-p)} of df2'
)

# 3.3. Multidimensional linear regression on df2 (mpg as target) with with train (5%) and test (95%)
lin_reg_train_df2_multidim = linear_regression_multidim(train_set_df2, x_cols_df2, y_col_df2)
lin_reg_test_df2_multidim = linear_regression_multidim(test_set_df2, x_cols_df2, y_col_df2)
# Predict and print the mpg for df2
mpg_predicted_train = train_set_df2[x_cols_df2].apply(lin_reg_train_df2_multidim, axis=1)
mpg_predicted_test = test_set_df2[x_cols_df2].apply(lin_reg_test_df2_multidim, axis=1)
print(f'mpg predicted train:\n{mpg_predicted_train}')
print(f'mpg predicted test:\n{mpg_predicted_test}')

# 3.4. Compute the mean squared error for the train set of df1 and df2
mse_train_set_df1 = mean_squared_error(train_set_df1, x_col_df1, y_col_df1, lin_reg_train_df1)
mse_train_set_df2 = mean_squared_error(train_set_df2, x_col_df2, y_col_df2, lin_reg_train_df2)

# 3.5. Compute the mean squared error for the test set of df1 and df2
mse_test_set_df1 = mean_squared_error(test_set_df1, x_col_df1, y_col_df1, lin_reg_test_df1)
mse_test_set_df2 = mean_squared_error(test_set_df2, x_col_df2, y_col_df2, lin_reg_test_df2)

# Compute the mean squared error for the train and test set of df2 (multidimensional)
mse_train_set_df2_multidim = mean_squared_error_multidim(
   train_set_df2,
   x_cols_df2,
   y_col_df2,
   lin_reg_train_df2_multidim
)

mse_test_set_df2_multidim = mean_squared_error_multidim(
   test_set_df2,
   x_cols_df2,
   y_col_df2,
   lin_reg_test_df2_multidim
)

# Print the mean squared error for the train and test sets of df1 and df2
print(f'''
MSE for the train set of df1: {mse_train_set_df1}
MSE for the test set of df1: {mse_test_set_df1}
MSE for the train set of df2: {mse_train_set_df2}
MSE for the test set of df2: {mse_test_set_df2}
MSE for the train set of df2 (multidimensional): {mse_train_set_df2_multidim}
MSE for the test set of df2 (multidimensional): {mse_test_set_df2_multidim}
''')

# 3.6. repeat n times with p% of the data
n = 100
p = np.random.random()
df1_sets = []
df2_sets = []

for i in range(n):
   df1_sets.append(split_data(df1, p))
   df2_sets.append(split_data(df2, p))

df1_train_sets = [subset[0] for subset in df1_sets]
df2_train_sets = [subset[0] for subset in df2_sets]
df1_test_sets = [subset[1] for subset in df1_sets]
df2_test_sets = [subset[1] for subset in df2_sets]

# Linear regression for df1 train sets
lin_reg_train_df1 = [linear_regression_one_dim_no_intercept(
   set,
   x_col_df1,
   y_col_df1
) for set in df1_train_sets]

# Linear regression for df1 test sets
lin_reg_test_df1 = [linear_regression_one_dim_no_intercept(
   set,
   x_col_df1,
   y_col_df1
) for set in df1_test_sets]

# Linear regression for df2 train sets
lin_reg_train_df2 = [linear_regression_one_dim(
   set,
   x_col_df2,
   y_col_df2
) for set in df2_train_sets]

# Linear regression for df2 test sets
lin_reg_test_df2 = [linear_regression_one_dim(
   set,
   x_col_df2,
   y_col_df2
) for set in df2_test_sets]

# Linear regression for df2 train sets (multidimensional)
lin_reg_train_df2_multidim = [linear_regression_multidim(
   set,
   x_col_df2,
   y_col_df2
) for set in df2_train_sets]

# Linear regression for df2 test sets (multidimensional)
lin_reg_test_df2_multidim = [linear_regression_multidim(
   set,
   x_col_df2,
   y_col_df2
) for set in df2_test_sets]

# Calculate the mean squared error for the train and test sets of df1 and df2
mse_train_set_df1 = [mean_squared_error(subset, x_col_df1, y_col_df1, lin_reg_train_df1[i]) for i, subset in enumerate(df1_train_sets)]
mse_train_set_df2 = [mean_squared_error(subset, x_col_df2, y_col_df2, lin_reg_train_df2[i]) for i, subset in enumerate(df2_train_sets)]
mse_test_set_df1 = [mean_squared_error(subset, x_col_df1, y_col_df1, lin_reg_test_df1[i]) for i, subset in enumerate(df1_test_sets)]
mse_test_set_df2 = [mean_squared_error(subset, x_col_df2, y_col_df2, lin_reg_test_df2[i]) for i, subset in enumerate(df2_test_sets)]

# Calculate the mean squared error for the train and test sets of df2 (multidimensional)
mse_train_set_df2_multidim = [mean_squared_error_multidim(subset, x_cols_df2, y_col_df2, lin_reg_train_df2_multidim[i]) for i, subset in enumerate(df2_train_sets)]
mse_test_set_df2_multidim = [mean_squared_error_multidim(subset, x_cols_df2, y_col_df2, lin_reg_test_df2_multidim[i]) for i, subset in enumerate(df2_test_sets)]

# Plot the mean squared errors
mse_histograms(
   [mse_train_set_df1, mse_train_set_df2, mse_test_set_df1, mse_test_set_df2, mse_train_set_df2_multidim, mse_test_set_df2_multidim],
   f'Histogram of the MSE over {n} iterations with a split of {percent_to_str(p)} - {percent_to_str(1-p)}',
   ['MSE for the train set of df1', 'MSE for the train set of df2', 'MSE for the test set of df1', 'MSE for the test set of df2', 'MSE for the train set of df2 (multidimensional)', 'MSE for the test set of df2 (multidimensional)'],
   'MSE',
   'Occurences'
)
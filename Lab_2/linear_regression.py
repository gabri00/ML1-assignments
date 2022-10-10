import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Set dataset path
PATH_DF_1 = 'turkish-se-SP500vsMSCI.csv'
PATH_DF_2 = 'mtcarsdata-4features.csv'

# Read the data from the csv file
df1 = pd.read_csv(PATH_DF_1)
df2 = pd.read_csv(PATH_DF_2)

# Remove spaces from the column names
df2.columns = df2.columns.str.strip()

# Linear regression one dimensional on df1
def linear_regression_one_dim(df, x_col, y_col):
      # Calculate mean of x and y
      x_mean = df[x_col].mean()
      y_mean = df[y_col].mean()
      # Calculate the numerator and denominator of the slope
      numerator = 0
      denominator = 0
      for x, y in zip(df[x_col], df[y_col]):
         numerator += (x - x_mean) * (y - y_mean)
         denominator += (x - x_mean) ** 2
      # Calculate the slope
      slope = numerator / denominator
      # Calculate the intercept
      intercept = y_mean - (slope * x_mean)
      # Return the linear regression function
      return lambda x: slope * x + intercept

# Plot the data and the linear regression line
def plot_linear_regression(
   df,
   x_col,
   y_col,
   linear_regression,
   xlabel,
   ylabel
   ):
      # Plot the data
      plt.scatter(df[x_col], df[y_col], marker='x')
      # Plot the linear regression line
      x = np.linspace(df[x_col].min(), df[x_col].max(), 100)
      plt.plot(x, linear_regression(x), color='red')

      plt.legend(['Linear Regression', 'Data'])
      plt.title(f'Linear Regression on {x_col} and {y_col}')
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
      
      plt.show()


# Columns to use from df1
x_col = 'SP500'
y_col = 'MSCI'

# Plot 1 configurations
plot1 = {
   'xlabel': f'American index {x_col}',
   'ylabel': f'UE index {y_col}'
}

# Select random subset of df1
subset = np.random.permutation(10)

# Plot the linear regression line for df1
plot_linear_regression(df1, x_col, y_col, linear_regression_one_dim(df1, x_col, y_col), plot1['xlabel'], plot1['ylabel'])

# Columns to use from df2
x_col = 'weight'
y_col = 'mpg'

# Plot 2 configurations
plot2 = {
   'xlabel': f'Car weight {x_col}',
   'ylabel': f'Fuel efficiency {y_col}'
}

# Plot the linear regression line for df2
plot_linear_regression(df2, x_col, y_col, linear_regression_one_dim(df2, x_col, y_col), plot2['xlabel'], plot2['ylabel'])
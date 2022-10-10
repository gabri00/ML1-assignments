import pandas as pd
import numpy as np

# Set dataset path
PATH_DF_1 = 'turkish-se-SP500vsMSCI.csv'
PATH_DF_2 = 'mtcarsdata-4features.csv'

# Read the data from the csv file
df1 = pd.read_csv(PATH_DF_1)
df2 = pd.read_csv(PATH_DF_2)

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
      # Return the slope and intercept
      return slope, intercept
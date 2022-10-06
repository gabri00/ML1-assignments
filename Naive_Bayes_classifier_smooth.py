# Naive Bayes
# 1. Read the data from the csv file
# 2. Convert categorical data to numerical data
# 3. Split the data into training and testing data
# 4. Remove the target column from the training and testing data
# 5. Build a Naive Bayes classifier with Laplace smoothing
# 6. Train the model
# 7. Predict the output
# 8. Print the output

import pandas as pd
import numpy as np

# Set dataset path
PATH = 'weather_dataset.csv'

# Read the data from the csv file
df = pd.read_csv(PATH, delim_whitespace=True)

# Set target column
TARG_COL = df.columns[-1]

# Set test and training proportions
TRAIN_SIZE = 10
TEST_SIZE = 4

# Set smoothing parameter
ALPHA = 1

# Convert categorical data to numerical data
def convert_categorical_to_numerical(df):
    for col in df.columns:
        class_map = {label: idx for idx, label in enumerate(np.unique(df[col]))}
        df[col] = df[col].map(class_map)

# Split the data into training and testing data
def split_data(df, train_size=TRAIN_SIZE, test_size=TEST_SIZE):
    df1 = df.sample(frac=1)
    train_X = df1.head(train_size)
    test_X = df1.tail(test_size)
    test_y = test_X[TARG_COL]
    # Remove the target column from the testing data
    test_X = test_X.drop(TARG_COL, axis=1)
    return train_X, test_X, test_y

# Count unique values in each column
def count_unique_values(df):
    unique_values = {}
    for col in df.drop(TARG_COL, axis=1).columns:
        unique_values[col] = df[col].value_counts().to_dict()
    return unique_values

# Build a Naive Bayes classifier with Laplace smoothing
class NaiveBayes:
    def fit(self, X, unique_values, alpha=ALPHA):
        self.X = X
        self.y = self.X[TARG_COL]
        self.classes = np.unique(self.y)
        self.unique_values = unique_values
        self.cond_prob = {}

        # Remove the target column from the training data
        self.X = self.X.drop(TARG_COL, axis=1)

        # Calculate the prior probability of the target column
        self.prior_prob = self.y.value_counts(normalize=True).to_dict()
        
        # Calculate the conditional probability of each feature
        for _, c in enumerate(self.classes):
            self.cond_prob[c] = {}
            X_c = self.X[self.y==c]
            for col in X_c.columns:
                c_counts = X_c[col].value_counts()
                for k in self.unique_values[col]:
                    if k not in c_counts:
                        c_counts[k] = self.unique_values[col][k]

                # Add smoothing
                self.cond_prob[c][col] = ((c_counts + alpha) / (X_c[col].value_counts().sum() + alpha*len(unique_values[col]))).to_dict()

    def predict(self, X):
        # Calculate the posterior probability for each pattern
        posterior = {}
        y = []
        for _, row in X.iterrows():
            for _, c in enumerate(self.classes):
                posterior[c] = self.prior_prob[c]
                for col in row.index:
                    posterior[c] *= self.cond_prob[c][col][row[col]]
            
            # for c in self.classes:
            #     posterior[c] /= (posterior[self.classes[0]] + posterior[self.classes[1]])

            y.append(self.classes[np.argmax(list(posterior.values()))])
        return y

    def score(self, y, y_pred):
        return np.sum(y_pred == y) / len(y)


convert_categorical_to_numerical(df)
train_X, test_X, test_y = split_data(df)
unique_values = count_unique_values(df)

model = NaiveBayes()
# Train the model
model.fit(train_X, unique_values)
# Predict the output
y_pred = model.predict(test_X)

# Print the output
print('True output: ', ["yes" if val else "no" for val in test_y.values])
print('Predicted output: ', ["yes" if val else "no" for val in y_pred])

# Calculate and print the accuracy of the model
print(f'Error: {int((1 - model.score(test_y, y_pred)) * 100)}%')
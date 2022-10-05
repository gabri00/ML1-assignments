# Naive Bayes
# 1. Read the data from the csv file
# 2. Convert categorical data to numerical data
# 3. Split the data into training and testing data
# 4. Remove the target column from the training and testing data
# 5. Build a Naive Bayes classifier
# 6. Train the model
# 7. Predict the output
# 8. Print the output

import pandas as pd
import numpy as np

# Read the data from the csv file
df = pd.read_csv('weather_dataset.csv', delim_whitespace=True)

# Convert categorical data to numerical data
for col in df.columns:
    class_map = {label: idx for idx, label in enumerate(np.unique(df[col]))}
    df[col] = df[col].map(class_map)

# Split the data into training and testing data
df1 = df.sample(frac=1)
train_X = df1.head(10)
test_X = df1.tail(4)
train_y = train_X['Play']
test_y = test_X['Play']

# Remove the target column from the training and testing data
train_X = train_X.drop('Play', axis=1)
test_X = test_X.drop('Play', axis=1)

# Add one to train_X and test_X to avoid zero probability
train_X = train_X + 1
test_X = test_X + 1

# Build a Naive Bayes classifier
class NaiveBayes:
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.cond_prob = {}

        # Calculate the prior probability of the target column
        self.prior_prob = y.value_counts(normalize=True).to_dict()
        
        # Calculate the conditional probability of each feature
        for _, c in enumerate(self.classes):
            self.cond_prob[c] = {}
            X_c = X[y==c]
            for col in X_c.columns:
                counts = X_c[col].value_counts()
                self.cond_prob[c][col] = (counts / counts.sum()).to_dict()

    def predict(self, X):
        # Calculate the posterior probability for each pattern
        posterior = {}
        y = []
        for row in range(len(X)):
            posterior[row] = {}
            for _, c in enumerate(self.classes):
                posterior[row][c] = self.prior_prob[c]
                for col in X.columns:
                    try:
                        posterior[row][c] *= self.cond_prob[c][col][X[col].iloc[row]]
                    except KeyError:
                        posterior[row][c] *= 0
            y.append(self.classes[np.argmax(list(posterior[row].values()))])

        return y

    def score(self, y, y_pred):
        return np.sum(y_pred == y) / len(y)

# Train the model
model = NaiveBayes()
model.fit(train_X, train_y)

# Predict the output
y_pred = model.predict(test_X)

# Print the output
print('True output: ', ["yes" if val else "no" for val in test_y.values])
print('Predicted output: ', ["yes" if val else "no" for val in y_pred])

# Calculate the accuracy of the model
print(f'Accuracy: {int(model.score(test_y, y_pred) * 100)}%')
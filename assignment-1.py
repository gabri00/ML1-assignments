import pandas as pd
import numpy as np

df = pd.read_csv('weather_dataset.csv', delim_whitespace=True)

for col in df.columns:
    class_map = {label: idx for idx, label in enumerate(np.unique(df[col]))}
    df[col] = df[col].map(class_map)

df1 = df.sample(frac=1)
train_set = df1.head(10)
test_set = df1.tail(4)
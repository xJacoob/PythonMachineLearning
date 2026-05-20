import itertools
import os

import numpy as np
import requests

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from matplotlib import pyplot as plt

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')

#data.info()
#print(data.head())

X = data.drop('salary', axis=1)
y = data['salary']

corr_matrix = X.corr(numeric_only=True)
labels = corr_matrix.values

most_correlated_columns = set()

for row in corr_matrix.index:
    for col in corr_matrix.columns:
        if corr_matrix.loc[row, col] > 0.2 and row != col:
            most_correlated_columns.add(col)

most_correlated_columns = list(most_correlated_columns)

combinations_single_column = list(itertools.combinations(most_correlated_columns, 1))
combinations_double_column = list(itertools.combinations(most_correlated_columns, 2))
combinations = list(combinations_single_column + combinations_double_column)

results = []

for col in combinations:
    X_temp = X.drop(columns=list(col))
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.3, random_state=100)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    prediction = lin_reg.predict(X_test)

    mape_error = mape(y_test, prediction)
    results.append((col, mape_error))

values = [value[1] for value in results]
lowest_mape_idx = np.argmin(values)
best_result = results[lowest_mape_idx]

X_new = X.drop(columns=list(best_result[0]))
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=100)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
prediction = lin_reg.predict(X_test)

replacement_with_zero = [value if value >= 0 else 0 for value in prediction]
replacement_with_median = [value if value >=0 else np.median(y_train) for value in prediction]

mapev1 = mape(y_test, replacement_with_zero)
mapev2 = mape(y_test, replacement_with_median)

print(f"{min(mapev1, mapev2):.5f}")


plt.imshow(corr_matrix, cmap='Accent')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
for a in range(labels.shape[0]):
    for b in range(labels.shape[1]):
        plt.text(b, a, '{:.2f}'.format(labels[a,b]), ha='center', va='center', color='black')
plt.show()

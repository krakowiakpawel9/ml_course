# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

'''
CRIM: Per capita crime rate by town
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX: Nitric oxide concentration (parts per 10 million)
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built prior to 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town
B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
LSTAT: Percentage of lower status of the population
MEDV: Median value of owner-occupied homes in $1000s
'''
# %% załadowanie danych
data_raw = load_boston()

# %% wstępne przetworzenie danych
df = pd.DataFrame(data_raw.data, columns=data_raw.feature_names)
df_target = pd.DataFrame(data_raw.target, columns=['Price'])

df = pd.concat([df, df_target], axis=1)

# %% sprawdzenie brakujących wartości
df.isnull().sum()

# %% eksploracyjna analiza danych (EDA)
cm = df.corr().round(2)
sns.heatmap(cm, cmap=sns.cm.rocket_r)

# %% przygotowanie danych do modelu
X = df.copy()
y = X.pop('Price')

# %% podział zbioru na zbiór treningowy i testowy
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# %% trenowanie modelu
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

# %% predykcja na podstawie modelu
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# %% ocena modelu na zbiorze treningowym
from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2 = r2_score(y_train, y_train_pred)

print('\nWydajność modelu na zbiorze treningowym:')
print('-' * 30)
print('RMSE:', rmse)
print('R2 score', r2, '\n')

# %% ocena modelu na zbiorze testowym
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print('\nWydajność modelu na zbiorze testowym:')
print('-' * 30)
print('RMSE:', rmse)
print('R2 score', r2, '\n')






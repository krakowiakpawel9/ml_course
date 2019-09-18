# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_decision_regions

sns.set()
np.random.seed(30)

# %% załadowanie danych
data_raw = load_iris()

# %% transformacja danych do obiektu DataFrame
df_data = pd.DataFrame(data=data_raw.data, columns=data_raw.feature_names)
df_target = pd.DataFrame(data=data_raw.target, columns=['target'])
df = pd.concat([df_data, df_target], axis=1)

# %% macierz korelacji
corr = df.corr()
print(corr['target'].sort_values(ascending=False))

# %% wybór dwóch zmiennych
df = df[['petal width (cm)', 'sepal length (cm)', 'target']]

# %% przygotowanie danych do modelu
X = df.iloc[:, :2]
y = df.iloc[:, [2]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train = X_train.values
y_train = y_train.values.ravel()
X_test = X_test.values
y_test = y_test.values.ravel()

# %% budowa modelu
classifier = SVC(kernel='rbf')

# %% trenowanie modelu
classifier.fit(X_train, y_train)

# %% Accuracy - dokładność, jak dobrze nasz model przewiduje klasy
# ocena modelu na zbiorze treningowym
acc_train = classifier.score(X_train, y_train)
print(f'Dokładność modelu na zbiorze treningowym: {acc_train}')

# %% ocena modelu na zbiorze testowym
acc_test = classifier.score(X_test, y_test)
print(f'Dokładność modelu na zbiorze testowym: {acc_test}')

# %% predykcja na podstawie modelu
y_pred = classifier.predict(X_test)

# %% raport klasyfikacji
print(classification_report(y_test, y_pred))

# %% macierz konfuzji
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %% wykreślenie granic decyzyjnych
plot_decision_regions(X_train, y_train, classifier)
plt.show()

# %% budowa funkcji 
def build_model(kernel='rbf'):

    classifier = SVC(kernel=kernel)
    classifier.fit(X_train, y_train)
    
    acc_train = classifier.score(X_train, y_train)
    print(f'Dokładność modelu na zbiorze treningowym: {acc_train}')

    acc_test = classifier.score(X_test, y_test)
    print(f'Dokładność modelu na zbiorze testowym: {acc_test}')

    y_pred = classifier.predict(X_test)

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    
    plot_decision_regions(X_train, y_train, classifier)
    plt.show()
    
    plt.figure()
    plot_decision_regions(X_test, y_test, classifier)
    plt.show()
    
# %%
build_model(kernel='rbf')

# %%
build_model(kernel='linear')    

# %%
build_model(kernel='poly')

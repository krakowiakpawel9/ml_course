# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sns.set()
np.random.seed(10)

# %% załadowanie danych
data_raw = load_iris()

# %% transformacja danych do obiektu DataFrame
df_data = pd.DataFrame(data=data_raw.data, columns=data_raw.feature_names)
df_target = pd.DataFrame(data=data_raw.target, columns=['target'])
df = pd.concat([df_data, df_target], axis=1)

names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['target_names'] = df.target.apply(lambda x: names[x])

# %% eksploracja danych
sns.pairplot(df, hue='target_names')

# %% pozostawienie dwóch klas versicolor i virginica
df = df[(df.target_names == 'versicolor') | (df.target_names == 'virginica')]

# %% eksploracja danych c.d.
sns.pairplot(df, hue='target_names')

# %% przygotowanie danych do modelu
X = df.iloc[:, :4]
y = df.iloc[:, 4:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# %% budowa modelu
classifier = SVC()

# %% trenowanie modelu
classifier.fit(X_train, y_train)

# %% Accuracy - dokładność, jak dobrze nasz model przewiduje klasy
# %% ocena modelu na zbiorze treningowym
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

# %% bez etykiet
sns.heatmap(cm, cmap=sns.cm.rocket_r, annot=True)

# %% z etykietami
cm = pd.DataFrame(cm, columns=['versicolor', 'virginica'], index=['versicolor', 'virginica'])
sns.heatmap(cm, cmap=sns.cm.rocket_r, annot=True)

# %% wykreślenie granic decyzyjnych
corr = df.corr()
print(corr['target'].sort_values(ascending=False))

# %%
df = df[['petal width (cm)', 'petal length (cm)', 'target']]
sns.pairplot(df, hue='target')

X = df.copy()
y = X.pop('target')




# %%
from mlxtend.plotting import plot_decision_regions

# plot_decision_regions(X_train, y_train, classifier)

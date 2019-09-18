# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
import pandas as pd
import seaborn as sns
sns.set()

pd.options.display.max_columns = 30

# %% załadowanie danych
data_raw = load_breast_cancer()
df_data = pd.DataFrame(data=data_raw.data, columns=data_raw.feature_names)
df_target = pd.DataFrame(data=data_raw.target, columns=['target'])
df = pd.concat([df_data, df_target], axis=1)

# %% sprawdzenie rozkładu klas
df.target.value_counts()

# %% podstawowe informacje
df.info()

# %% podstawowe statystyki
df.describe()

# %% sprawdzenie brakujących wartości
df.isnull().sum()

# %% przygotowanie danych do modelu
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# %% podział na zbiór treningowy i testowy, podział warstowy
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

print(y.value_counts() / len(y))
print(y_train.value_counts() / len(y_train))
print(y_test.value_counts() / len(y_test))

# %% budowa modelu
from sklearn.svm import SVC

classifier = SVC(C=1.0, kernel='linear')
classifier.fit(X_train, y_train)

# %% ocena modelu na zbiorze treningowym
acc_train = classifier.score(X_train, y_train)
print(f'Dokładność modelu na zbiorze treningowym: {acc_train}')

# %% ocena modelu na zbiorze testowym
acc_test = classifier.score(X_test, y_test)
print(f'Dokładność modelu na zbiorze testowym: {acc_test}')

# %% predykcja na podstawie modelu
y_pred = classifier.predict(X_test)

# %% raport klasyfikacji
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))

# %% macierz konfuzji
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %% bez etykiet
sns.heatmap(cm, cmap=sns.cm.rocket_r, annot=True)

# %% budowa modelu 
from sklearn.svm import SVC

classifier = SVC(C=1.0, kernel='linear')
classifier.fit(X_train, y_train)

# %% ocena modelu na zbiorze treningowym
acc_train = classifier.score(X_train, y_train)
print(f'Dokładność modelu na zbiorze treningowym: {acc_train}')

# %% ocena modelu na zbiorze testowym
acc_test = classifier.score(X_test, y_test)
print(f'Dokładność modelu na zbiorze testowym: {acc_test}')

# %% predykcja na podstawie modelu
y_pred = classifier.predict(X_test)

# %% raport klasyfikacji
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))

# %% macierz konfuzji
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %% bez etykiet
sns.heatmap(cm, cmap=sns.cm.rocket_r, annot=True)
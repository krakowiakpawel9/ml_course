# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
np.random.seed(10)
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

# %% krzywa ROC - Receiver Operating Characteristic
from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve(y_test, y_pred)

def plot_roc_curve(fpr, tpr, label=None):
    plt.figure()
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('FPR - False Positive Rate')
    plt.ylabel('TPR - True Positive Rate')
    plt.show()

plot_roc_curve(fpr, tpr)

# %% C parametr
for C in [0.1, 0.5]:
    classifier = SVC(C=C, kernel='linear')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    plot_roc_curve(fpr, tpr)
    
# %% 

def evaluate_binary_classification(classifier,
                                   X_train=X_train,
                                   X_test=X_test,
                                   y_train=y_train,
                                   y_test=y_test):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import roc_curve
    
    sns.set()
    
    acc_train = classifier.score(X_train, y_train)
    print(f'Dokładność modelu na zbiorze treningowym: {acc_train:.4f}')

    acc_test = classifier.score(X_test, y_test)
    print(f'Dokładność modelu na zbiorze testowym: {acc_test:.4f}')

    y_pred = classifier.predict(X_test)
    
    print(f'\nRaport klasyfikacji:\n {classification_report(y_test, y_pred)}')

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    sns.heatmap(cm, cmap=sns.cm.rocket_r, annot=True)
    plt.show()
    
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    
    def plot_roc_curve(fpr, tpr, label=None):
        plt.figure()
        plt.plot(fpr, tpr, label=label)
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel('FPR - False Positive Rate')
        plt.ylabel('TPR - True Positive Rate')
        plt.title('ROC curve')
        plt.show()
    
    plot_roc_curve(fpr, tpr)
    
# %%
evaluate_binary_classification(classifier)
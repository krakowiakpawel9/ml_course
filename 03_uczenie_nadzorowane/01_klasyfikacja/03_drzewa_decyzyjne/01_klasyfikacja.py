# -*- coding: utf-8 -*-
# %% import bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
pd.options.display.max_columns = 30
np.random.seed(10)
# %% zdefiniowane funkcje


def evaluate_binary_classification(classifier, X_train, X_test, y_train, y_test):
    
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
    
    
# %% pobranie danych
url = ('https://ml-repository-krakers.s3-eu-west-1.amazonaws.com/ml_course/'
       'datasets/hr_dataset/general_data.csv')

df = pd.read_csv(url)

# %%
df.info()

# %% 
df.describe()

# %%
df.describe(include=['object'])

# %%
print(df.isnull().sum())

# %%
df.dropna(inplace=True)

# %%
df.Attrition = df.Attrition.apply(lambda x: 1 if x == 'Yes' else 0)

# %% 
for col in df.columns:
    if len(df[col].unique()) < 2:
        print(col)

# %%
df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)
df.columns = [col.replace('&', 'and') for col in df.columns]       

# %%
corr = df.corr()
print(corr.Attrition.sort_values(ascending=False))
sns.heatmap(corr, cmap=sns.cm.rocket_r)
plt.show()

# %%
df = pd.get_dummies(df, drop_first=True)

# %%
X = df.copy()
y = X.pop('Attrition')

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# %%
print(y_train.value_counts() / len(y_train))
print(y_test.value_counts() / len(y_test))

# %% 
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(X_train, y_train)

# %% 
evaluate_binary_classification(classifier, X_train, X_test, y_train, y_test)

# %%
# sudo apt install python-pydot python-pydot-ng graphviz
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,
               filled=True, rounded=True,
               special_characters=True,
               feature_names=X.columns,
               class_names=['Yes', 'No'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('graph.png')
Image(graph.create_png())

# %%
from sklearn.model_selection import GridSearchCV

param_grid = [{'max_depth': range(1, 20),
              'min_samples_leaf': [1, 2, 3, 4, 5, 10, 15],
              'criterion': ['gini', 'entropy']}]

model = DecisionTreeClassifier()    
    
gs = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)
    
# %%
evaluate_binary_classification(gs, X_train, X_test, y_train, y_test)   

# %%
print(gs.best_params_)

# %% zapisanie modelu do pliku, pickle
import pickle

model = gs.best_estimator_
with open('./models/18092019_decision_tree_classifier.pickle', 'wb') as file:
    pickle.dump(model, file)

# %% zapisanie modelu do pliku, joblib
import joblib

with open('./models/18092019_decision_tree_classifier.pkl', 'wb') as file:
    joblib.dump(model, file)

# %% eksport X_test, y_test
X_test.to_csv('./datasets/X_test.csv')
y_test.to_csv('./datasets/y_test.csv', header='target')









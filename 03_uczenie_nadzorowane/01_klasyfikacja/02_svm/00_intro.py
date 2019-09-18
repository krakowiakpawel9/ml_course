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

names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['target_names'] = df.target.apply(lambda x: names[x])
# %%
sns.relplot(data=df.query('target == 0 or target == 1'),
            x='petal length (cm)',
            y='sepal width (cm)',
            hue='target_names', sizes=100)
plt.plot([1, 5], [2, 4], c='red')
plt.show()

# %%
X = df[['petal length (cm)', 'sepal width (cm)', 'target']].query('target == 0 or target == 1')
y = X.pop('target')
X = X.values
y = y.values

# %%
classifier = SVC(kernel='rbf', C=1.0)
classifier.fit(X, y)

# %%
plot_decision_regions(X, y, classifier)
plt.show()

# %%
ax = plot_decision_regions(X, y, classifier)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['setosa', 'virginica'], framealpha=0.3)

# %%
plt.scatter(np.linspace(-0.4, 0.7, 11), np.linspace(-0.4, 0.7, 11) * 0)
plt.scatter(np.linspace(-1.5, -0.4, 11), np.linspace(-1.5, -0.4, 11) * 0, c='orange')
plt.scatter(np.linspace(0.7, 1, 6), np.linspace(0.7, 1, 6) * 0, c='orange')

# %%
plt.scatter(np.linspace(-0.4, 0.7, 11), np.linspace(-0.4, 0.7, 11) ** 2)
plt.scatter(np.linspace(-1.5, -0.4, 11), np.linspace(-1.5, -0.4, 11) ** 2, c='orange')
plt.scatter(np.linspace(0.7, 1, 6), np.linspace(0.7, 1, 6) ** 2, c='orange')








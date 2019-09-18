# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns

sns.set()
pd.options.display.max_columns = 30
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

# %%
corr = df.corr()
print(corr.Attrition.sort_values(ascending=False))
sns.heatmap(corr, cmap=sns.cm.rocket_r)

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

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# %% 
classifier.score(X_test, y_test)

# %%
# pip install graphviz
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













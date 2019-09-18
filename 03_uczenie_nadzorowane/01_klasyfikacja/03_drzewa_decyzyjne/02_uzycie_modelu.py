# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import joblib

# %% załadowanie danych
X_test = pd.read_csv('./datasets/X_test.csv', index_col=0)
y_test = pd.read_csv('./datasets/y_test.csv', index_col=0)

# %% załadowanie modelu, pickle
with open('./models/18092019_decision_tree_classifier.pickle', 'rb') as file:
    model = pickle.load(file)

# %% ocena modelu
print(model.score(X_test, y_test))

# %% załadowanie modelu, joblib
with open('./models/18092019_decision_tree_classifier.pkl', 'rb') as file:
    model = joblib.load(file)

# %%
print(model.score(X_test, y_test))    
   
    
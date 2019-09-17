# -*- coding: utf-8 -*-
import pandas as pd

url = ('https://ml-repository-krakers.s3-eu-west-1.amazonaws.com/ml_course/'
       'datasets/Churn_Modelling.csv')
df = pd.read_csv(url, index_col=0)

# %%
url2 = ('https://ml-repository-krakers.s3-eu-west-1.amazonaws.com/ml_course/'
        'datasets/car_price/train-data.csv')
df2 = pd.read_csv(url2, index_col=0)

# %%
url3 = ('https://ml-repository-krakers.s3-eu-west-1.amazonaws.com/ml_course/'
        'datasets/Mall_Customers.csv')
df3 = pd.read_csv(url3, index_col=0)
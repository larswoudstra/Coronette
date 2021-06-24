import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression
from tensorflow.keras import layers



############

# load the data
covid_df_train = pd.read_csv("data_covid/covid.train.csv")
covid_df_test = pd.read_csv("data_covid/covid.test.csv")

# remove id-column from dataframe (not a feature)
covid_df_train = covid_df_train.drop(['id'], axis=1)
covid_df_test = covid_df_test.drop(['id'], axis=1)

# split the dataframe into data and labels
train_data_df = covid_df_train.iloc[:, :-1]
test_data_df = covid_df_test

# create a column with the means for the behavioural indicators for each day
behaviour_avg_col_tr = train_data_df.loc[:, 'wearing_mask':'public_transit'].mean(axis=1)
behaviour_avg_col_1_tr = train_data_df.loc[:, 'wearing_mask.1':'public_transit.1'].mean(axis=1)
behaviour_avg_col_2_tr = train_data_df.loc[:, 'wearing_mask.2':'public_transit.2'].mean(axis=1)

# create a column with the means for the behavioural indicators on all days
behaviour_avg = pd.concat([behaviour_avg_col_tr, behaviour_avg_col_1_tr, behaviour_avg_col_2_tr], axis=1).mean(axis=1)

train_data_df.drop(['worried_become_ill', 'worried_finances', 'anxious.1', 'depressed.1',
'felt_isolated.1', 'worried_become_ill.1', 'worried_finances.1', 'anxious.2',
'depressed.2', 'felt_isolated.2', 'worried_become_ill.2', 'worried_finances.2'], axis=1)

test_data_df.drop(['worried_become_ill', 'worried_finances', 'anxious.1', 'depressed.1',
'felt_isolated.1', 'worried_become_ill.1', 'worried_finances.1', 'anxious.2',
'depressed.2', 'felt_isolated.2', 'worried_become_ill.2', 'worried_finances.2'], axis=1)

train_data_df.insert(44, "Mean behaviour", 0)

print(train_data_df.iloc[:, 42:46].head())

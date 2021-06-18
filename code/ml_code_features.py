import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, f_regression

# load the data
covid_df = pd.read_csv("data_covid/covid.train.csv")

# remove id-column from dataframe (not a feature)
covid_df = covid_df.drop(['id'], axis=1)

# split the dataframe into data and labels
data = covid_df.iloc[:, :-1]
targets = covid_df.iloc[:, -1:]

# transform dataframe to numpy arrays
data = data.to_numpy()
targets = targets.to_numpy()

data_new = SelectKBest(f_regression, k="all").fit_transform(X, y)

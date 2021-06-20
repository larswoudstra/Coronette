import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, f_regression

# load the data
covid_df_train = pd.read_csv("data_covid/covid.train.csv")
covid_df_test = pd.read_csv("data_covid/covid.test.csv")

# remove id-column from dataframe (not a feature)
covid_df_train = covid_df_train.drop(['id'], axis=1)
covid_df_test = covid_df_test.drop(['id'], axis=1)

# split the dataframe into data and labels
train_data = covid_df_train.iloc[:, :-1]
train_targets = covid_df_train.iloc[:, -1:]

# transform dataframes to numpy arrays
train_data = train_data.to_numpy()
train_targets = train_targets.to_numpy()

test_data = covid_df_test.to_numpy()

def select_features(X_train, y_train, X_test, k={}):
    """ Determines the features with the highest importance, based on correlation with the output. """

    # select all features
    feature_scores = SelectKBest(f_regression, k=k)

	# run the score function on the training data to determine the best features
    feature_scores.fit(X_train, y_train)

	# transform training and test data into a selection of the best features
    X_train_best = feature_scores.transform(X_train)
    X_test_best = feature_scores.transform(X_test)

    return X_train_best, X_test_best, feature_scores

train_data_all, test_data_all, feature_scores = select_features(train_data, train_targets, test_data, k="all")

# # create a bar plot for the feature scores to determine what to set k as in SelectKBest
# x = np.arange(len(feature_scores.scores_))
# plt.bar(x, feature_scores.scores_)
# plt.xlabel("Features")
# plt.ylabel("Feature score")
# plt.show()

train_data_best, test_data_best, feature_scores = select_features(train_data, train_targets, test_data, k=14)
print(train_data_best)

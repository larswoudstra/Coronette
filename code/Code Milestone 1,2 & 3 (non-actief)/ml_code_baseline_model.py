import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

########################################
# Part 1: loading and cleaning the data

# load the data
covid_df_train = pd.read_csv("data_covid/covid.train.csv")
covid_df_test = pd.read_csv("data_covid/covid.test.csv")

# remove id-column from dataframe (not a feature)
covid_df_train = covid_df_train.drop(['id'], axis=1)
covid_df_test = covid_df_test.drop(['id'], axis=1)

# split the dataframe into data and target values
train_data_df = covid_df_train.iloc[:, :-1]
train_targets_df = covid_df_train.iloc[:, -1:]

# transform dataframes to numpy arrays
train_data = train_data_df.to_numpy()
train_targets = train_targets_df.to_numpy()

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

# select 'k' best features based on barplot, see images: 'best_features_barplot'
k = 93

train_k_best, test_k_best, feature_scores = select_features(train_data, train_targets.ravel(), test_data, k=k)

# split the data into training and validation data
train_data, val_data, train_targets, val_targets = train_test_split(train_k_best, train_targets,
                                                    train_size=0.7, random_state=14)

########################################
# Part 2: creating the model

# we start by creating a simple neural network
import tensorflow as tf
from tensorflow.keras import layers, models, metrics

# function that creates a neural network with:
# - 78 input nodes
# - 1 hidden layer (78 nodes, reLU activation)
# - 1 output node
def build_neural_net():
    # initialize the model
    model = models.Sequential()

    # add layers
    model.add(layers.Dense(units=k, activation='relu', input_shape=(k,)))

    # end with two output units
    model.add(layers.Dense(units=1))

    # calculate the accuracy of the model ##### mean_squared_error als loss?
    model.compile(loss='mean_squared_error', optimizer='Nadam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

########################################
# Part 3: training the model

# initialize model
model = build_neural_net()

# train model
history = model.fit(train_data, train_targets, epochs=300, validation_data=(val_data, val_targets))

########################################
# Part 4: evaluating the model

# plot the training loss and validation loss defined by RMSE
train_loss = history.history['root_mean_squared_error']
val_loss = history.history['val_root_mean_squared_error']
plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['train_loss', 'val_loss'])
plt.title('RMSE losses')
plt.show()

print(f"Validation RMSE: {model.evaluate(val_data, val_targets)[1]}")

# calculate the differences between predicted and real data
y_pred = model.predict(val_data)
difference = y_pred - val_targets

plt.plot(difference, color='red')
plt.title('Difference')
plt.show()

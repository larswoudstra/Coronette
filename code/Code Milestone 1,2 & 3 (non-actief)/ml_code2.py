import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

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
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> 3a2e0b6a7bd2b1fcecca2fbe13041fa0db6a4251
k = 93
=======
k = 14
>>>>>>> 5751b6e7d4adcd8f2a3deeed1f55106243878268

train_k_best, test_k_best, feature_scores = select_features(train_data, train_targets.ravel(), test_data, k=k)

# Part 2: creating and testing the model

# implement k-fold cross validation
kf = KFold(5, shuffle = True)

rmse_val = 0
rmse_train = 0

fold = 0
for train, val in kf.split(train_k_best, train_targets):
    fold += 1
    print(f'Fold #{fold}')

    train_data_fold = train_k_best[train]
    train_targets_fold = train_targets[train]

    val_data_fold = train_k_best[val]
    val_targets_fold = train_targets[val]

    normalizer = preprocessing.Normalization()
    normalizer.adapt(train_data_fold)
    normalizer.adapt(val_data_fold)

    # for each fold, initialize a neural network
    model = models.Sequential()

    # add fully connected layers
    # - 93 input nodes
    # - 3 hidden layers (93, 60, and nodes, reLU activation)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    model.add(layers.Dense(units=(k*(2/3)), activation='relu', input_shape=(k,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(units=round(k*(1/3)), activation='relu'))
    model.add(layers.BatchNormalization())
=======
    model.add(layers.Dense(units=k, activation='relu', input_shape=(k,)))
>>>>>>> 3a2e0b6a7bd2b1fcecca2fbe13041fa0db6a4251
=======
=======
    model.add(layers.Dense(units=k, activation='relu', input_shape=(k,)))
<<<<<<< HEAD
>>>>>>> c422bf231096aa361cf626dc3cd5609bc0f43f2e
    model.add(layers.Dense(units=5, activation='relu', input_shape=(k,)))
>>>>>>> 5751b6e7d4adcd8f2a3deeed1f55106243878268
=======
    model.add(layers.Dense(units=5, activation='relu'))
>>>>>>> c006d0bd033a530a39e8f29070e95972ce1838c2

    # - 1 output node with a linear activation function
    model.add(layers.Dense(units=1))

    # compile the model with the Nadam optimizer
    model.compile(loss='mean_squared_error', optimizer='Nadam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # train the model
    history = model.fit(train_data_fold, train_targets_fold, batch_size=70, epochs=1500, validation_data=(val_data_fold, val_targets_fold))

    y_pred = model.predict(val_data_fold)

    rmse_train += np.asarray(history.history['root_mean_squared_error'])
    rmse_val += np.asarray(history.history['val_root_mean_squared_error'])

# Part 4: model evaluation

# calculate average RMSE
rmse_train_avg = rmse_train / fold
rmse_val_avg = rmse_val / fold

# plot the average RMSE
plt.plot(rmse_train_avg)
plt.plot(rmse_val_avg)
plt.legend(['RMSE train', 'RMSE val'])
plt.show()

# evaluate the model
print(f"Training RMSE: {model.evaluate(train_data_fold, train_targets_fold)[1]}")
print(f"Validation RMSE: {model.evaluate(val_data_fold, val_targets_fold)[1]}")

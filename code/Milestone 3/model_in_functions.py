import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression


######################################## Part 1: loading and cleaning the data

def load_data(set):
    # load the data
    covid_df = pd.read_csv(f"data_covid/covid.{set}.csv")
    # remove id-column
    covid_df = covid_df.drop(['id'], axis=1)

    return covid_df

def transform_data(training_data):
    # get data and target values
    data_df = training_data.iloc[:, :-1]
    targets_df = training_data.iloc[:, -1:]

    # transform into numpy arrays
    data = data_df.to_numpy()
    targets = targets_df.to_numpy()

    return data, targets

# load training and testing datasets
covid_df_train = load_data("train")
covid_df_test = load_data("test")

# get training data and training targets
train_data, train_targets = transform_data(covid_df_train)

# transform testing data into numpy
test_data = covid_df_test.to_numpy()


# select k best features from data
def select_features(X_train, y_train, X_test, k={}):
    """Determines the features with the highest importance, based on correlation with the output."""

    # select all features
    feature_scores = SelectKBest(f_regression, k=k)

	# run the score function on the training data to determine the best features
    feature_scores.fit(X_train, y_train)

	# transform training and test data into a selection of the best features
    X_train_best = feature_scores.transform(X_train)
    X_test_best = feature_scores.transform(X_test)

    return X_train_best, X_test_best, feature_scores

# select 'k' best features based on barplot (see 'best_features_barplot')
train_k_best, test_k_best, feature_scores = select_features(train_data, train_targets.ravel(), test_data, k=14)



######################################## Part 2: creating and testing the model

# implement k-fold cross validation
kf = KFold(5, shuffle=True)

# init variables that contain the calculated RMSE
rmse_val = 0
rmse_train = 0

# variable that contains the current fold
fold = 0

# train the model using different training and validation sets (k-fold)
for train, val in kf.split(train_k_best, train_targets):

    # keep track of current fold
    fold += 1
    print(f'Fold #{fold}')

    # get training data and targets from data
    train_data_fold = train_k_best[train]
    train_targets_fold = train_targets[train]

    # get validation data and targets from data
    val_data_fold = train_k_best[val]
    val_targets_fold = train_targets[val]


    # initialize a neural network
    model = models.Sequential()

    # add fully connected layers
    # - 14 (k) input nodes
    # - 2 hidden layers (14x5, relu-activation)
    # - 1 output layer (1, linear activation)
    model.add(layers.Dense(units=14, activation='relu', input_shape=(14,)))
    model.add(layers.Dense(units=5, activation='relu'))
    model.add(layers.Dense(units=1))

    # compile the model with the Nadam optimizer
    model.compile(loss='mean_squared_error', optimizer='Nadam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # train the model
    history = model.fit(train_data_fold, train_targets_fold, batch_size=70,
    epochs=700, validation_data=(val_data_fold, val_targets_fold))

    # get prediction
    y_pred = model.predict(val_data_fold)

    # compute RMSE-values for training and validation data
    rmse_train += np.asarray(history.history['root_mean_squared_error'])
    rmse_val += np.asarray(history.history['val_root_mean_squared_error'])



###################################################### Part 4: model evaluation

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
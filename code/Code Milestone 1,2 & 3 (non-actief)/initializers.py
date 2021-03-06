import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression


# Part 1: loading and cleaning the data

def load_data(set, n):
    """Loads csv-file into a dataframe and removes the first column (ID).
    Creates a test dataframe out of every nth row of the complete dataset"""
    # load the data
    covid_df = pd.read_csv(f"data_covid/covid.{set}.csv")
    # remove id-column
    covid_df = covid_df.drop(['id'], axis=1)

    # select every nth row out of full train data set to create test data
    test_df = covid_df.iloc[::n]

    # remove test data from training data
    train_df = covid_df.drop(covid_df.index[::n])

    return train_df, test_df

def transform_data(training_data):
    """Splits dataset into data and target values. Transforms data from dataframe
    to array."""
    # get data and target values
    data_df = training_data.iloc[:, :-1]
    targets_df = training_data.iloc[:, -1:]

    # transform into numpy arrays
    data = data_df.to_numpy()
    targets = targets_df.to_numpy()

    return data, targets

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


# Part 2: creating and testing the model

def train_neural_network(train_data_fold, train_targets_fold, val_data_fold, val_targets_fold):
    """Creates and trains a neural network. Returns the history."""
    # set the 'He' weight initializer
    initializer = tf.keras.initializers.he_normal(seed=None)

    # initialize a neural network
    model = models.Sequential()

    # add fully connected layers
    model.add(layers.Dense(units=5, activation='relu', input_shape=(14,), kernel_initializer=initializer))
    model.add(layers.Dense(units=1))

    # compile the model with the Nadam optimizer
    model.compile(loss='mean_squared_error', optimizer='Nadam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # train the model
    history = model.fit(train_data_fold, train_targets_fold, batch_size=70, epochs=700, validation_data=(val_data_fold, val_targets_fold))

    # get predictions
    preds = model.predict(val_data_fold)

    # evaluate the model
    print(f"Training RMSE: {model.evaluate(train_data_fold, train_targets_fold)[1]}")
    print(f"Validation RMSE: {model.evaluate(val_data_fold, val_targets_fold)[1]}")

    return history, preds

def get_data_and_targets(train, val, training_data, training_targets):
    """Splits training data and targets into training and validation data."""
    return training_data[train], training_targets[train], training_data[val], training_targets[val]


def kfold_NN(train_k_best, train_targets):
    """Runs neural network and applies k-fold cross validation. Returns the
    RMSE values for the training and validation data."""
    # init variables that contain the calculated RMSE
    rmse_val = 0
    rmse_train = 0
    # variable that contains the current fold
    fold = 0

    # implement k-fold cross validation
    kf = KFold(5, shuffle=True)

    # train the model using different training and validation sets (k-fold)
    for train, val in kf.split(train_k_best, train_targets):

        # keep track of current fold
        fold += 1
        print(f'Fold #{fold}')

        # get training data and targets from data
        train_data_fold, train_targets_fold, val_data_fold, val_targets_fold = get_data_and_targets(train, val, train_k_best, train_targets)

        history, preds = train_neural_network(train_data_fold, train_targets_fold, val_data_fold, val_targets_fold)

        # compute RMSE-values for training and validation data
        rmse_train += np.asarray(history.history['root_mean_squared_error'])
        rmse_val += np.asarray(history.history['val_root_mean_squared_error'])

    plot_RMSE(rmse_train, rmse_val)


# Part 4: model evaluation

def plot_RMSE(rmse_train, rmse_val, fold=5):
    """Plots the average RMSE for the training and validation data."""
    # calculate average RMSE
    rmse_train_avg = rmse_train / fold
    rmse_val_avg = rmse_val / fold

    # plot the average RMSE
    plt.plot(rmse_train_avg)
    plt.plot(rmse_val_avg)
    plt.legend(['RMSE train', 'RMSE val'])
    plt.title(f'The RMSE validation value is: {rmse_val_avg[-1]:.2f}')
    plt.show()

def plot_differences(y_preds, y_targets):
    differences = y_preds - y_targets

    plt.hist(differences, bins = 100)
    plt.title('Histogram of differences between prediction values and target values')
    plt.show()

def test_NN(train_data, n, k):
    """Creates a test data set out of the full training dataframe and tests the
    trained model"""
    # select every nth row out of full train data set to create test data
    test_df = train_data.iloc[1::n]
    test_data, test_targets = transform_data(test_df)

    # remove test data from training data
    train_df = train_data.drop(train_data.index[1::n])
    train_data, train_targets = transform_data(train_df)

    #select features
    train_k_best, test_k_best, feature_scores = select_features(train_data, train_targets.ravel(), test_data, k=14)

    # train the model
    history, predictions = train_neural_network(train_k_best, train_targets, test_k_best, test_targets)

    # compute RMSE-values for training and validation data
    rmse_train = np.asarray(history.history['root_mean_squared_error'])
    rmse_val = np.asarray(history.history['val_root_mean_squared_error'])

    plot_RMSE(rmse_train, rmse_val, fold=1)

    plot_differences(predictions, test_targets)


# CODE FOR WEIGHT INITIALIZERS

def train_neural_network_initializer(train_data_fold, train_targets_fold, val_data_fold, val_targets_fold, initializers):
    """Creates and trains a neural network. Returns the history."""

    loop = 0

    histories = []
    predictions = []

    for initializer in initializers:

        loop += 1

        # initialize a neural network
        model = models.Sequential()

        # add fully connected layers
        model.add(layers.Dense(units=5, activation='relu', input_shape=(14,), kernel_initializer=initializer))
        model.add(layers.Dense(units=1))

        # compile the model with the Nadam optimizer
        model.compile(loss='mean_squared_error', optimizer='Nadam',
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])

        # train the model
        history = model.fit(train_data_fold, train_targets_fold, batch_size=40, epochs=400, validation_data=(val_data_fold, val_targets_fold))
        histories.append(history)

        # get predictions
        preds = model.predict(val_data_fold)
        predictions.append(preds)

        # evaluate the model
        # print(f"Validation RMSE: {model.evaluate(val_data_fold, val_targets_fold)[1]}")

    return histories, predictions


def kfold_NN_initializers(train_k_best, train_targets, initializers):
    """Runs neural networks with different initializers and applies
    k-fold cross validation. Returns the RMSE values for the validation
    data for each of the initializers."""

    # init variables that contain the calculated RMSE
    rmse_val_he = 0
    rmse_val_random_normal = 0
    rmse_val_glorot_uniform = 0

    # variable that contains the current fold
    fold = 0

    # implement k-fold cross validation
    kf = KFold(7, shuffle=True)

    # train the model using different training and validation sets (k-fold)
    for train, val in kf.split(train_k_best, train_targets):

        # keep track of current fold
        fold += 1
        print(f'Fold #{fold}')

        # get training data and targets from data
        train_data_fold, train_targets_fold, val_data_fold, val_targets_fold = get_data_and_targets(train, val, train_k_best, train_targets)

        histories, predictions = train_neural_network_initializer(train_data_fold, train_targets_fold, val_data_fold, val_targets_fold, initializers)

        loop = 0

        for history, pred in zip(histories, predictions):
            loop += 1

        # compute RMSE-values for each initializer:
        # - He initializer
            if loop == 1:
                rmse_val_he += np.asarray(history.history['val_root_mean_squared_error'])

        # - RandomNormal initializer
            if loop == 2:
                rmse_val_random_normal += np.asarray(history.history['val_root_mean_squared_error'])

        # - Glorot (Xavier) initializer
            if loop == 3:
                rmse_val_glorot_uniform += np.asarray(history.history['val_root_mean_squared_error'])

    # calculate average RMSE
    rmse_val_he_avg = rmse_val_he / fold
    rmse_val_random_normal_avg = rmse_val_random_normal / fold
    rmse_val_glorot_uniform_avg = rmse_val_glorot_uniform / fold

    # plot the average RMSE
    plt.plot(rmse_val_he_avg)
    plt.plot(rmse_val_random_normal_avg)
    plt.plot(rmse_val_glorot_uniform_avg)
    plt.legend(['He', 'RandomNormal', 'Glorot (Xavier)'])
    plt.title(f'The RMSE validation values for three initializers')
    plt.show()

    print(f'Validation RMSE for He initializer: {rmse_val_he_avg[-1]:.2f}')
    print(f'Validation RMSE for RandomNormal initializer: {rmse_val_random_normal_avg[-1]:.2f}')
    print(f'Validation RMSE for Glorot (Xavier) initializer: {rmse_val_glorot_uniform_avg[-1]:.2f}')

# Run program

if __name__ == "__main__":
    # load training and testing datasets
    covid_df_train, covid_df_test = load_data("train", 5)

    # get training data and training targets
    train_data, train_targets = transform_data(covid_df_train)

    # transform testing data into numpy
    test_data, test_targets = transform_data(covid_df_test)

    # select 'k' best features based on barplot (see 'best_features_barplot')
    train_k_best, test_k_best, feature_scores = select_features(train_data, train_targets.ravel(), test_data, k=14)

    # Set initializers
    initializers = [tf.keras.initializers.he_normal(seed=None),
                    tf.keras.initializers.RandomNormal(mean=0.,stddev=1.),
                    tf.keras.initializers.GlorotUniform()]


    # # train neural network using k-fold cross validation
    kfold_NN_initializers(train_k_best, train_targets, initializers)

    # test the neural network creating train and test data
    # test_NN(covid_df_train, 5, 14)

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

def train_neural_network(train_data_fold, train_targets_fold, val_data_fold, val_targets_fold, batch_size, epochs, hidden_layer_sizes):
    """Creates and trains a neural network. Returns the history."""

    # set the 'He' weight initializer
    initializer = tf.keras.initializers.he_normal(seed=None)

    # initialize a neural network
    model = models.Sequential()

    # define the input size
    input_size = train_data_fold.shape[1]

    # add first fully connected layer with a ReLU-activation function and an input size
    model.add(layers.Dense(units=hidden_layer_sizes[0], activation='relu', input_shape=(input_size,), kernel_initializer=initializer))

    # add hidden layers with ReLU-activation function(s)
    for layer in hidden_layer_sizes[1:-1]:
        model.add(layers.Dense(units=layer, activation='relu'))

    # add output layer without activation function
    model.add(layers.Dense(units=1))

    # compile the model with the Nadam optimizer
    model.compile(loss='mean_squared_error', optimizer='Nadam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # train the model
    history = model.fit(train_data_fold, train_targets_fold, batch_size=batch_size, epochs=epochs, validation_data=(val_data_fold, val_targets_fold))

    # get predictions
    preds = model.predict(val_data_fold)

    # evaluate the model
    print(f"Training RMSE: {model.evaluate(train_data_fold, train_targets_fold)[1]}")
    print(f"Validation RMSE: {model.evaluate(val_data_fold, val_targets_fold)[1]}")

    return history, preds

def get_data_and_targets(train, val, training_data, training_targets):
    """Splits training data and targets into training and validation data."""
    return training_data[train], training_targets[train], training_data[val], training_targets[val]


def kfold_NN(train_k_best, train_targets, batch_size, epochs, hidden_layer_sizes, folds):
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

        history, preds = train_neural_network(train_data_fold, train_targets_fold, val_data_fold, val_targets_fold, batch_size=batch_size, epochs=epochs, hidden_layer_sizes=hidden_layer_sizes)

        # compute RMSE-values for training and validation data
        rmse_train += np.asarray(history.history['root_mean_squared_error'])
        rmse_val += np.asarray(history.history['val_root_mean_squared_error'])

    plot_RMSE(rmse_train, rmse_val)

# Part 4: model evaluation

def plot_RMSE(rmse_train, rmse_val, fold):
    """Plots the average RMSE for the training and validation data."""

    # calculate average RMSE
    rmse_train_avg = rmse_train / fold
    rmse_val_avg = rmse_val / fold

    # plot the average RMSE
    plt.plot(rmse_train_avg)
    plt.plot(rmse_val_avg)
    plt.legend(['RMSE train', 'RMSE val'])
    plt.title(f'The RMSE validation value is: {rmse_val[-1]:.2f}')
    plt.show()

def plot_differences(y_preds, y_targets):
    differences = y_preds - y_targets

    plt.hist(differences, bins = 100)
    plt.title('Histogram of differences between prediction values and target values')
    plt.show()

def NN(train_data, train_targets, test_data, test_targets, k_features=14, batch_size=70, epochs=700, hidden_layer_sizes=[5], kfold=False, folds=5):
    """Creates a test data set out of the full training dataframe and tests the
    trained model"""

    # select features
    train_k_best, test_k_best, feature_scores = select_features(train_data, train_targets.ravel(), test_data, k=k_features)

    if kfold == True:
        kfold_NN(train_k_best, train_targets, batch_size=batch_size, epochs=epochs, hidden_layer_sizes=hidden_layer_sizes, folds=folds)

    else:
        # train the model
        history, predictions = train_neural_network(train_k_best, train_targets, test_k_best, test_targets, batch_size=batch_size, epochs=epochs, hidden_layer_sizes=hidden_layer_sizes)

        # compute RMSE-values for training and validation data
        rmse_train = np.asarray(history.history['root_mean_squared_error'])
        rmse_val = np.asarray(history.history['val_root_mean_squared_error'])

        plot_RMSE(rmse_train, rmse_val, fold=1)

        plot_differences(predictions, test_targets)

# Run program

if __name__ == "__main__":
    # load training and testing datasets
    covid_df_train, covid_df_test = load_data("train", 5)

    # get training data and training targets
    train_data, train_targets = transform_data(covid_df_train)

    # transform testing data into numpy
    test_data, test_targets = transform_data(covid_df_test)

    # test the neural network creating train and test data
    NN(train_data, train_targets, test_data, test_targets, batch_size=70, epochs=700, hidden_layer_sizes=[5], k_features=14, kfold=False, folds=5)

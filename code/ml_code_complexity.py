import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# Part 1: loading and cleaning the data

# load the data
covid_df = pd.read_csv("data_covid/covid.train.csv")

# remove id-column from dataframe (not a feature)
covid_df = covid_df.drop(['id'], axis=1)

# split the dataframe into data and the corresponding targets
data = covid_df.iloc[:, :-1]
targets = covid_df.iloc[:, -1:]

# transform dataframe to numpy arrays
data = data.to_numpy()
targets = targets.to_numpy()


# Part 2: creating and testing the model

import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from sklearn.model_selection import KFold

# implement k-fold cross validation
kf = KFold(5, shuffle = True)

rmse_val = 0
rmse_train = 0

fold = 0
for train, val in kf.split(data):
    fold += 1
    print(f'Fold #{fold}')

    train_data = data[train]
    train_targets = targets[train]

    val_data = data[val]
    val_targets = targets[val]

    # for each fold, initialize a neural network
    model = models.Sequential()

    # add fully connected layers
    # - 93 input nodes
    # - 2 hidden layers (93 nodes, reLU activation)
    model.add(layers.Dense(units=93, activation='relu', input_shape=(93,)))
    model.add(layers.Dense(units=93, activation='relu'))

    # - 1 output node with a linear activation function
    model.add(layers.Dense(units=1))

    # compile the model with the ... optimizer
    model.compile(loss='mean_squared_error', optimizer='Nadam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # train the model
    history = model.fit(train_data, train_targets, epochs=300, validation_data=(val_data, val_targets))

    y_pred = model.predict(val_data)

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
print(f"Training RMSE: {model.evaluate(train_data, train_targets)[1]}")
print(f"Validation RMSE: {model.evaluate(val_data, val_targets)[1]}")

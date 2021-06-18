import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

########################################
# Part 1: loading and cleaning the data

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

# split the data into training and validation data
# train_data, val_data, train_targets, val_targets = train_test_split(data, targets,
#                                                     train_size=0.7, random_state=14)

########################################
# Part 2: creating the model

# we start by creating a simple neural network
import tensorflow as tf
from tensorflow.keras import layers, models, metrics
from sklearn.model_selection import KFold

# function that creates a neural network with:
# - 93 input nodes
# - 1 hidden layer (93 nodes, reLU activation)
# - 1 output node

# cross validation
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


    # initialize the model
    model = models.Sequential()

    # add layers
    model.add(layers.Dense(units=93, activation='relu', input_shape=(93,)))

    # end with two output units
    model.add(layers.Dense(units=1))

    # compile the model
    model.compile(loss='mean_squared_error', optimizer='Adamax',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # train model
    history = model.fit(train_data, train_targets, epochs=300, validation_data=(val_data, val_targets))

    y_pred = model.predict(val_data)

    # print(history.history['root_mean_squared_error'])

    rmse_train += np.asarray(history.history['root_mean_squared_error'])
    rmse_val += np.asarray(history.history['val_root_mean_squared_error'])

# calculate average
rmse_train_avg = rmse_train / fold
rmse_val_avg = rmse_val / fold

plt.plot(rmse_train_avg)
plt.plot(rmse_val_avg)
plt.legend(['RMSE train', 'RMSE val'])
plt.show()


print(f"Training RMSE: {model.evaluate(train_data, train_targets)[1]}")
print(f"Validation RMSE: {model.evaluate(val_data, val_targets)[1]}")

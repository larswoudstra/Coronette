import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Part 1: loading and cleaning the data

# load the data
covid_df = pd.read_csv("data_covid/covid.train.csv")
# print(covid_df.head())

# remove id-column from dataframe
covid_df = covid_df.drop(['id'], axis=1)

# check for missing values ### if-statement weghalen?
if covid_df.isnull().values.any():
    covid_df = covid_df.dropna()

# split the dataframe into data and labels
data = covid_df.iloc[:, :-1]
labels = covid_df.iloc[:, -1:]

data = data.to_numpy()
labels = labels.to_numpy()

# # select a subset of the state columns
# state_columns = data.iloc[:, :40]
# # create a new column with the one-hot encoded vectors
# data["state"] = state_columns.apply(lambda r: tuple(r), axis=1).apply(np.array)
# # remove the state columns
# data = data.drop(data.iloc[:, :40], axis=1)

# we moeten nog ff die laatste kolom naar voren brengen **************

# split the data into training and validation data
train_data, val_data, train_labels, val_labels = train_test_split(data, labels,
                                                    train_size=0.7, random_state=14)

###############################################################################
# Part 2: creating the model

# we start by creating a simple neural network
from tensorflow import keras
from tensorflow.keras import layers, models

# function that creates a neural network with:
# - 54 input nodes
# - 1 hidden layer (54 nodes, reLU activation)
# - 2 output nodes (softmax activation)
def build_neural_net():
    # initialize the model
    model = models.Sequential()

    # add layers
    model.add(layers.Dense(units=93, activation='relu', input_shape=(93,)))

    # end with two output units
    model.add(layers.Dense(units=1))

    # calculate the accuracy of the model ##### mean_squared_error als loss?
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


###############################################################################
# Part 3: training the model

# initialize model
model = build_neural_net()

# train model ######## grootte batch? hier doen ze validation split?
history = model.fit(train_data, train_labels, epochs=500)

# # retrieve loss and accuracy of the model
# loss, accuracy = model.evaluate(val_data, val_labels)
#
# # Print to 3 decimals
# print(f'Test loss: {loss:.3}')


###############################################################################
# Part 4: evaluating the model

y_pred = model.predict(val_data)

plt.plot(val_labels, color = 'red', label = 'Real data')
plt.plot(y_pred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

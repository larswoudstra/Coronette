import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Part 1: loading and cleaning the data

# load the data
covid_df = pd.read_csv("data_covid/covid.train.csv")
# print(covid_df.head())

# remove id-column from dataframe (not a feature)
covid_df = covid_df.drop(['id'], axis=1)

# check for missing values ### if-statement weghalen?
if covid_df.isnull().values.any():
    covid_df = covid_df.dropna()

# split the dataframe into data and labels
data = covid_df.iloc[:, :-1]
labels = covid_df.iloc[:, -1:]

# transform dataframe to numpy arrays
data = data.to_numpy()
labels = labels.to_numpy()

# split the data into training and validation data
train_data, val_data, train_labels, val_labels = train_test_split(data, labels,
                                                    train_size=0.7, random_state=14)


# NOTE: use MinMax scaler to normalize data?

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
    model.add(layers.Dense(units=1, activation='linear'))

    # calculate the accuracy of the model ##### mean_squared_error als loss?
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    return model


###############################################################################
# Part 3: training and evaluating the model

# initialize model
model = build_neural_net()

# train model
history = model.fit(train_data, train_labels, epochs=500)

# retrieve loss and accuracy of the model
loss, accuracy  = model.evaluate(val_data, val_labels)

# Print to 3 decimals
print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')

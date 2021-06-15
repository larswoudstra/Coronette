# test code

import pandas as pd
from sklearn.model_selection import train_test_split

# load the data
covid_df = pd.read_csv("data_covid/covid.train.csv")
# print(covid_df.head())

covid_df.drop(['id'], axis=1)

# check if there are any missing values
if not covid_df.isnull().values.any():
    continue

# split the data on data and labels
data = covid_df.iloc[:, :-1]
labels = covid_df.iloc[:, -1:]

train_data, val_data, train_labels, val_labels = train_test_split(data, labels,
                                                    train_size=0.7, random_state=14)

# we start by creating a simple neural network
from tensorflow import keras
from tensorflow.keras import layers, models

# initialize the model
model = models.Sequential()

# add layers
model.add(layers.Dense(units=94, activation='relu', input_shape=(94,)))

# end with two output units
model.add(layers.Dense(units=2, activation='softmax'))

# calculate the accuracy of the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

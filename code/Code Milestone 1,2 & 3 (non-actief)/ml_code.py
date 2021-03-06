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

# remove missing values
covid_df = covid_df.dropna()

# remove mental health features to prevent overfitting
covid_df = covid_df.drop(['anxious', 'depressed', 'felt_isolated',
'worried_become_ill', 'worried_finances', 'anxious.1', 'depressed.1',
'felt_isolated.1', 'worried_become_ill.1', 'worried_finances.1', 'anxious.2',
'depressed.2', 'felt_isolated.2', 'worried_become_ill.2', 'worried_finances.2'],
axis=1)

# split the dataframe into data and labels
data = covid_df.iloc[:, :-1]
targets = covid_df.iloc[:, -1:]

# transform dataframe to numpy arrays
data = data.to_numpy()
targets = targets.to_numpy()

# split the data into training and validation data
train_data, val_data, train_targets, val_targets = train_test_split(data, targets,
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
    model.add(layers.Dense(units=78, activation='relu', input_shape=(78,)))

    # end with two output units
    model.add(layers.Dense(units=1))

    # calculate the accuracy of the model ##### mean_squared_error als loss?
    model.compile(loss='mean_squared_error', optimizer='adam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

########################################
# Part 3: training the model

# initialize model
model = build_neural_net()

# train model
<<<<<<< HEAD
history = model.fit(train_data, train_targets, epochs=500, validation_data=(val_data, val_targets))
=======
history = model.fit(train_data, train_targets, epochs=50, validation_data=(val_data, val_targets))
>>>>>>> 7bc86c114719a3cd2e89ba03ff394da1d781690b

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

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, metrics

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
train_data, val_data, train_targets, val_targets = train_test_split(data, targets,
                                                    train_size=0.7, random_state=14)

########################################
# Part 2: creating the model

# function that creates a neural network with:
# - 93 input nodes
# - 1 hidden layer (93 nodes, reLU activation)
# - 1 output node
def build_neural_net():

    # initialize the model
    model = models.Sequential()

    # add hidden layer
    model.add(layers.Dense(units=93, activation='relu', input_shape=(93,)))

    # add output layer
    model.add(layers.Dense(units=1))

    # calculate the loss of the model
    model.compile(loss='mean_squared_error', optimizer='adam',
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

########################################
# Part 3: training the model

# initialize model
model = build_neural_net()

# train model
history = model.fit(train_data, train_targets, epochs=500, validation_data=(val_data, val_targets))

########################################
# Part 4: evaluating the model

# calculate the differences between predicted and real data
y_pred = model.predict(val_data)

# plot the predicted data and real data to see differences
plt.plot(val_targets, color='red', label='Real data')
plt.plot(y_pred, color='blue', alpha=0.3, label='Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

# plot the differences between predicted and real data
difference = y_pred - val_targets
plt.plot(difference, color='red')
plt.title('Difference')
plt.show()

# plot the training loss and validation loss defined by RMSE
train_loss = history.history['root_mean_squared_error']
val_loss = history.history['val_root_mean_squared_error']
plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['train_loss', 'val_loss'])
plt.show()

print(f"Validation RMSE: {model.evaluate(val_data, val_targets)[1]}")

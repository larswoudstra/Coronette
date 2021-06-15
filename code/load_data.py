import pandas as pd

# load the data
covid_df = pd.read_csv("data_covid/covid.train.csv")
# print(covid_df.head())

# check if there are any missing values
if not covid_df.isnull().values.any():
    print("Yay")

from sklearn.model_selection import train_test_split

# split the data on data and labels
data = covid_df.iloc[:, :-1]
labels = covid_df.iloc[:, -1:]

train_data, val_data, train_labels, val_labels = train_test_split(data, labels,
                                                    train_size=0.7, random_state=14)

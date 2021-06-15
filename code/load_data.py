import pandas as pd

# load the data
covid_train_data = pd.read_csv("/Users/ninaalblas/Documents/Minor AI/ML Project/Coronette/ml2021spring-hw1 (1)/covid.train.csv")
print(covid_train_data.head())

# check if there are any missing values
if not covid_train_data.isnull().values.any():
    print("Yay")

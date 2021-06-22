# Feature importance
14_14_7_1
Hoge training RMSE en hoge val RMSE
Val

# Feature importance
14 inputs met 3 hidden layers:
14, 9, 5, 1
Hoge training RMSE en hoge val RMSE
Dit lijken simpelweg te veel hidden layers voor zo weinig features

# Feature importance
53 inputs (Alle staten weggelaten) met 2 hidden layers: (53, 53, 27, 1)
- Train RMSE: 0.91
- Val RMSE: 0.99

# Dropout
3 dropout layers van 0.2, 0.2 en 0.3 voor layer 3, 4 en 5 bij k = 93 (alle features)
Hoge training RMSE en hoge vval RMSE

# Batch size
Alle 93 features en een batch size van 20 met 3 hidden layers (93, 93, 60, 30, 1)
- Train RMSE: 0.8716
- Val RMSE: 0.9780

# Feature importance
78 features, mental health features met de hand weggeselecteerd
Batch size = 20
3 hidden layers: 78, 78, 52, 26, 1
- Train RMSE: 1.01
- Val RMSE: 1.07

# Baseline model
Baseline model zonder kfold, 93 features, 1 hidden layer (93, 93, 1)
Geen batch batch size
Plot met heeeeeeel veel pieken 

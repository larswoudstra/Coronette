# VANDAAG
* Code nagelopen en wat aanpassingen gemaakt in een nieuwe file 'ml_code_complexity' aan o.a. de comments en stukjes code
  * train_test_split code weggehaald (we gebruiken nu k-fold cross validation dus niet meer nodig)
  * Op sommige plekken stond er nog 'labels' in plaats van 'targets'
* Complexiteit verhogen op het model wat we tot nu toe hebben:
  * Fully connected (Dense)
  * Alle features (93 totaal)
  * 1 hidden layer met 93 hidden units en ReLu activatie functie
  * 1 output unit met lineaire activatie functie
  * Nadam optimizer
  * Loss: MSE; metrics: RMSE
    * K-fold cross validated


Wat we moeten doen:
- analyse van beste features
  -- sklearn.feature_selection.SelectKBest(score_func=<function f_classif>, *, k=10)
      f_regression of mutual_info_regression

      f_regression : F-value between label/feature for regression tasks.

- feature importance aanpassen

Use "all" --> We can then print the scores for each variable (largest is better) and plot the scores for each variable as a bar graph to get an idea of how many features we should select.

In de barplot is te zien dat er 2 features evident belangrijker waren dan de andere variabelen in de training data. Gezien de index van de pieken lijken dit dezelfde variabelen te zijn, maar gemeten op dag 2 en dag 3. Hetzelfde geldt voor een viertal variabelen dat voor iedere gemeten dag een redelijke correlatie vertoont. Gezien de 14 pieken, wordt SelectKBest uitgevoerd met k=14 om de beste features te selecteren.






We zijn er mee bezig gegaan, maar toen zijn we er achter gekomen dat er na de
verbetering van de validatiedata geen sprake meer was van overfitting. Om die
reden hebben we het kijken naar de features wat naar achter op het to-do-lijstje
gezet.

If you use sparse data (i.e. data represented as sparse matrices), chi2, mutual_info_regression, mutual_info_classif will deal with the data without making it dense.

SelectFromModel is a meta-transformer that can be used alongside any estimator that assigns importance to each feature through a specific attribute (such as coef_, feature_importances_) or via an importance_getter callable after fitting. The features are considered unimportant and removed if the corresponding importance of the feature values are below the provided threshold parameter. Apart from specifying the threshold numerically, there are built-in heuristics for finding a threshold using a string argument. Available heuristics are “mean”, “median” and float multiples of these like “0.1*mean”. In combination with the threshold criteria, one can use the max_features parameter to set a limit on the number of features to select.

Correlation is a measure of how two variables change together. Perhaps the most common correlation measure is Pearson’s correlation that assumes a Gaussian distribution to each variable and reports on their linear relationship.

The scikit-learn machine library provides an implementation of the correlation statistic in the f_regression() function.

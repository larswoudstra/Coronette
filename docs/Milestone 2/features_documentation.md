Wat we moeten doen:
- analyse van beste features
  -- sklearn.feature_selection.SelectKBest(score_func=<function f_classif>, *, k=10)
      f_regression of mutual_info_regression

      f_regression : F-value between label/feature for regression tasks.

- feature importance aanpassen

Use "all" --> We can then print the scores for each variable (largest is better) and plot the scores for each variable as a bar graph to get an idea of how many features we should select.

In de barplot is te zien dat er 2 features evident belangrijker waren dan de andere variabelen in de training data. Gezien de index van de pieken lijken dit dezelfde variabelen te zijn, maar gemeten op dag 2 en dag 3. Hetzelfde geldt voor een viertal variabelen dat voor iedere gemeten dag een redelijke correlatie vertoont. Gezien de 14 pieken, wordt SelectKBest uitgevoerd met k=14 om de beste features te selecteren.

De 14 features met de hoogste scores zijn ('tested_positive.1', 148069.65827795214), ('tested_positive', 69603.87259124119), ('hh_cmnty_cli', 9235.492094457415), ('hh_cmnty_cli.1', 9209.019557848724), ('hh_cmnty_cli.2', 9097.375171712854), ('nohh_cmnty_cli', 8395.42129963222), ('nohh_cmnty_cli.1', 8343.255927399816), ('nohh_cmnty_cli.2', 8208.176434709772), ('cli', 6388.906849111197), ('cli.1', 6374.547999948064), ('cli.2', 6250.008701728293), ('ili', 5998.92288019529), ('ili.1', 5937.588576440179), ('ili.2', 5796.94767178646))

Alle variabelen (behalve de staten) zijn percentages, en dus op dezelfde schaal. De correlaties hoeven dus niet genormaliseerd te worden.
*Klopt dit?*
*En staten meenemen?*

Nu de irrelevante variabelen verwijderen of kun je ze ook minder belangrijk maken?
**Ik heb Wouter gemaild.**

What are Benefits of performing feature selection before modeling your data?
· Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.
· Improves Accuracy: Less misleading data means modeling accuracy improves.
· Reduces Training Time: fewer data points reduce algorithm complexity and algorithms train faster.




Correlation is a measure of how two variables change together. Perhaps the most common correlation measure is Pearson’s correlation that assumes a Gaussian distribution to each variable and reports on their linear relationship.

The scikit-learn machine library provides an implementation of the correlation statistic in the f_regression() function.

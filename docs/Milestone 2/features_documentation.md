Om overfitting tegen te gaan, kan er een selectie van features gemaakt worden. Deze *feature selection* helpt niet alleen overfitting te verminderen, het zou ook de kosten en trainingstijd moeten terugdringen. Hiervoor moet worden onderzocht welke variabelen de beste voorspellers zijn voor de target feature (een positieve testuitslag). Dit wordt gedaan door middel van sklearn's SelectKBest-functie. Hierbij moet een k, ofwel het aantal features met de hoogste scores, gedefinieerd worden.

Eerst wordt de SelectKBest-functie uitgevoerd met alle 93 features (k="all"). De hieruit volgende correlatiescores worden in een barplot gevisualizeerd.

![bar plot](https://github.com/larswoudstra/Coronette/blob/main/docs/images/best_features_barplot.png)

De hoeveelheid features met de hoogste pieken geeft een idee van de benodigde waarde van k. Hieruit blijken 14 features met de hoogste correlatiescore.

![best 14 features](https://github.com/larswoudstra/Coronette/blob/main/docs/images/best_14_features.png)

Het gaat hier dus eigenlijk slechts om een viertal features (hh_cmnty_cli, nohh_cmnty_cli, cli en ili) dat voor ieder van de drie gemeten dagen een relatief hoge correlatie vertoont, en de features 'tested_positive' van dag 1 en 2. Deze laatste zijn verantwoordelijk voor de hoogste pieken in de grafiek en lijken daarmee de sterkste voorspellers te zijn voor de positieve testuitslag op dag 3.

Gezien de 14 pieken wordt SelectKBest nogmaals uitgevoerd, maar nu met k=14 waarna er een nieuwe training- en testdataset gevormd wordt met de 14 features met de hoogste correlatiescores.



<!-- f_regression : F-value between label/feature for regression tasks.

Alle variabelen (behalve de staten) zijn percentages, en dus op dezelfde schaal. De correlaties hoeven dus niet genormaliseerd te worden.
*Klopt dit?*
*En staten meenemen?*

Correlation is a measure of how two variables change together. Perhaps the most common correlation measure is Pearson’s correlation that assumes a Gaussian distribution to each variable and reports on their linear relationship.

The scikit-learn machine library provides an implementation of the correlation statistic in the f_regression() function. -->

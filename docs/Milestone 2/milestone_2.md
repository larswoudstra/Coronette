# Data analysis
Zie voor voorgaande data-analyse [Milestone1](https://github.com/larswoudstra/Coronette/blob/main/docs/Milestone%201/Milestone_1.md).

## Feature selection
Aan de hand van de RMSE learning curve van het eerste model leek het in de eerste instantie er op dat het model aan het overfitten was. Dit werd geconcludeerd uit de spikes die de gemaakte loss functie had:

< afbeelding 'losses_plus_mental_health'>

In veel van de gevallen heeft de validation data een hoge loss. Dit betekent dat het model de percentages van de validation data niet goed kan voorspellen. Om dit probleem te verhelpen is er geprobeerd om met minder features te gaan werken. De mental health features leken hiervoor het best geschikt om weg te laten, omdat deze het minste de daadwerkelijke percentages zouden kunnen voorspellen **Beter onderbouwen**. De features die zijn weggelaten zijn: anxious, depressed, felt_isolated, worried_become_ill en worried_finances. Dit had het volgende resultaat:

< afbeelding 'losses_min_mental_health'>

In de plot is te zien dat de amplitude van de spikes iets is afgenomen, maar de spikes zijn nog steeds aanwezig. Aangezien dit betekent dat het model in sommige epochs nog altijd overfit en in andere niet, is ervoor gekozen de selectie van features beter te onderbouwen, namelijk middels een SelectKBest-analyse. Hierbij is de scorefunctie 'f_regression' gebruikt, omdat deze voorkomt dat er noise wordt toegevoegd.

Dergelijke *feature selection* helpt niet alleen overfitting te verminderen, het zou ook de kosten en trainingstijd moeten terugdringen. Hiervoor moet worden onderzocht welke variabelen de beste voorspellers zijn voor de target feature (het percentage positieve testuitslagen). Bij sklearn's SelectKBest-functie moet een k, ofwel het aantal features met de hoogste scores, gedefinieerd worden.

Eerst wordt de SelectKBest-functie uitgevoerd met alle 93 features (k="all"). De hieruit volgende correlatiescores worden in een barplot gevisualiseerd.

![bar plot](https://github.com/larswoudstra/Coronette/blob/main/docs/images/best_features_barplot.png)

De hoeveelheid features met de hoogste pieken geeft een idee van de benodigde waarde van k. Hieruit blijken 14 features met de hoogste pieken, en dus de hoogste correlatiescore. Deze worden dus sowieso meegenomen. Na de evaluatie wordt er gekeken welke hoeveelheid features daadwerkelijk tot het beste resultaat leidt.

![best 14 features](https://github.com/larswoudstra/Coronette/blob/main/docs/images/best_14_features.png)

Met deze 14 features gaat het dus vooralsnog eigenlijk slechts om een viertal features (hh_cmnty_cli, nohh_cmnty_cli, cli en ili) dat voor ieder van de drie gemeten dagen een relatief hoge correlatie vertoont, en de features 'tested_positive' van dag 1 en 2. Deze laatste zijn verantwoordelijk voor de hoogste pieken in de grafiek en lijken daarmee de sterkste voorspellers te zijn voor de positieve testuitslag op dag 3.

Gezien de 14 pieken wordt SelectKBest nogmaals uitgevoerd, maar nu met k=14 waarna er een nieuwe training- en testdataset gevormd wordt met de 14 features met de hoogste correlatiescores. Gekeken naar de modelprestaties kan dit aantal nog aangepast worden. Mocht het model met deze 14 features bijvoorbeeld juist underfitten, kan dit probleem opgelost worden door meer features toe te voegen.

# Data Pipeline

## K-fold cross validation
In Milestone 1 is er gebruikgemaakt van de train_test_split-functie van sklearn om de data te verdelen in 70% trainingdata en 30% validatiedata.

<losses_plus_mental_health.png>

Er wordt verwacht dat de pieken in de grafiek veroorzaakt worden door een te kleine validatieset, die bovendien niet representatief is voor de data waarvoor het model getraind is. Om dit op te lossen is er gebruik gemaakt van K-fold cross validation met shuffle en 5 folds.   

## Complexiteit verhogen
Er zijn verschillende combinaties en groottes van hidden layers getraind. Bij een model van 93 features blijkt een architectuur van 93x93x60x1 tot de laagste validatiekoste te komen (RMSE = 0.98). Bij een model van 14 features ligt de optimale opzet van het netwerk bij 14x7x1 (RMSE = 0.95). Om deze kosten nog verder terug te dringen, wordt er gekeken naar andere hoeveelheden features, Dropout en BatchNormalization.

**!!!!!**

# Model training

## K-fold cross validation
Voor iedere fold is een model getraind met alle features en optimizer adam om het effect van de K-fold te laten zien ten opzichte van het baseline model van Milestone 1.

## Optimizer
De validatiedata heeft in veel van de gevallen een hoge loss. Daarom wordt er gezocht naar een betere optimizer voor het Neural Network. De eerste optimizer die in Milestone 1 is gebruikt is de 'adam' optimizer. Om te bepalen welke optimizer het beste past bij de data, zijn alle optimizers van Keras geprobeerd met verschillende learning rates. Om te bepalen welke optimizer met welke learning rate het beste bij het model past, wordt de RMSE loss opnieuw geplot.

Om een eerste indruk te krijgen van de verschillende optimizers en deze goed te kunnen vergelijken wordt er in eerste instantie gewerkt met 100 epochs. Een analyse van iedere mogelijke optimizer met 100 epochs en ieders default learning rate - 0.01 voor SGD en  0.001 voor de overige optimizers - wijst een aantal interessante dingen uit. Zo blijken SGD (RMSE = 7.81), Adam (RMSE = 1.27), Adadelta (RMSE = 3.26) en Adagrad (RMSE = 2.20) niet de beste keuzes te zijn, omdat deze modellen al vrij snel in het trainingproces (bijvoorbeeld na 20 epochs) al niet verder lijken te leren met nog relatief hoge kosten.

Adamax (RMSE = 1.08), Nadam (RMSE = 1.07), RMSProp (RMSE = 1.13) en Ftrl (RMSE = 1.24) blijken na de analyse met 100 epochs daarentegen nog wel optimizers die mogelijk het overwegen waard zijn, omdat deze de laagste validatiekosten opleveren en/of nog duidelijk aan het leren zijn tegen het einde van het trainen.

Na een analyse met 300 epochs en nog altijd ieders default learning rate lijkt Nadam (RMSE = 0.94) de beste keuze te zijn voor het regressieprobleem. Bij RMSProp stegen de validatiekosten namelijk naarmate het aantal epochs toenam (RMSE = 1.37). Ftrl (RMSE = 1.23) leek niet veel meer te leren met 200 extra epochs, maar de kostenplot was wel erg glad. Hoewel Adamax het ook zeker niet slecht doet (RMSE = 0.98), lijkt dit model al gestopt te zijn met leren waar Nadams validatiekosten nog dalen. Bovendien resulteerde het model met de Nadam optimizer in de laagste kosten, en deze lijken zelfs nog een beetje te dalen na 300 epochs.

Al met al lijkt de Nadam optimizer de beste keuze, maar Adamax en Ftrl moeten in het achterhoofd gehouden worden, aangezien deze mogelijk betere resultaten geven met andere learning rates, extra hidden layers of extra hidden nodes.

Er wordt dus nog steeds full batch gradient descent gebruikt, maar dan nu met de 'Nadam'-optimizer. Er worden 300 epochs gebruikt, omdat dit aantal voldoende informatie lijkt te geven over de werking van het model.

# Model evaluation

## K-fold cross validation
Om het model te kunnen evalueren is de RMSE van alle 5 folds gemiddeld genomen. Dit is vervolgens geplot zoals eerder is gedaan in Milestone 1. Voor alle 5 folds zijn 300 epochs gebruikt. Het verschil in learning curves is te zien in de volgende plots. De eerste plot is de learning curve van het baseline model, de tweede plot is de learning curve van het model met K-fold cross validation.

< afbeelding 'Learning_curve_K_fold_all_features' >

Hieruit is te concluderen dat de spikes vooral te maken hebben gehad met niet representatieve validatie data, in plaats van met overfitting wat in de eerste instantie werd gedacht. De RMSE van dit model is nog steeds niet optimaal, dus wordt er verder gekeken naar het opimaliseren van het model doormiddel van het toevoegen van complexiteit. Zie *complexiteit verhogen*

## RMSE Metric
Om te bepalen hoe goed het model nu daadwerkelijk is, wordt er gebruik gemaakt van een RootMeanSquaredError learning curve. Deze plot geeft het gemiddelde verschil tussen de daadwerkelijke waarden en de voorspelde waarden; bij een hoge RMSE is het verschil groot, bij een kleine RMSE is het verschil klein.

## Difference plot
Wanneer een voorspeld percentage lager is dan het daadwerkelijke percentage positieve Covid tests, kan dit grotere gevolgen hebben dan wanneer het percentage hoger is voorspeld. Om dit in kaart te brengen is er een difference plot aan het model toegevoegd. Deze plot laat zien wat de verschillen zijn tussen de voorspelde percentages en de daadwerkelijke percentages van de validation data:

< afbeelding 'difference_plot'

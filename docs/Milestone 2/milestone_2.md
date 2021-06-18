# Data analysis

## RMSE Metric
Om te bepalen hoe goed het model nu daadwerkelijk is, wordt er gebruik gemaakt van een RootMeanSquaredError learning curve. Deze plot geeft het gemiddelde verschil tussen de daadwerkelijke waarden en de voorspelde waarden; bij een hoge RMSE is het verschil groot, bij een kleine RMSE is het verschil klein.

## Feature removal
Aan de hand van de RMSE learning curve van het eerste model leek het in de eerste instantie er op dat het model aan het overfitten was. Dit werd geconcludeerd uit de spikes die de gemaakte loss functie had:

< afbeelding 'losses_plus_mental_health'>

In veel van de gevallen heeft de validation data een hoge loss. Dit betekent dat het model de percentages van de validation data niet goed kan voorspellen. Om dit probleem te verhelpen is er geprobeerd om met minder features te gaan werken. De mental health features leken hiervoor het best geschikt om weg te laten, omdat deze het minste de daadwerkelijke percentages zouden kunnen voorspellen. De features die zijn weggelaten zijn: anxious, depressed, felt_isolated, worried_become_ill en worried_finances. Dit had het volgende resultaat:

< afbeelding 'losses_min_mental_health'>

In de plot is te zien dat de amplitude van de spikes iets is afgenomen, maar de spikes zijn nog steeds aanwezig. Omdat deze features wel enigszins invloed zouden kunnen hebben op het percentage van positieve Covid tests, lijkt het beter om de features niet helemaal uit de data te verwijderen, maar om deze features minder zwaar te laten wegen.

## Feature importance
Stuk over *feature importance*

# Data Pipeline

## K-fold cross validation
In milestone 1 is er gebruik gemaakt van de train_test_split functie van sklearn om de data te verdelen in 70% training data en 30% validation data. Hierdoor is de validatie data niet representatief voor de data waarvoor het model getraind is. In de eerste instantie werd er gedacht dat de spikes werden veroorzaakt door overfitting, maar dit bleek niet het echte probleem te zijn. De spikes werden veroorzaakt door te weinig validatiedata die ook niet representatief is. Om dit op te lossen is er gebruik gemaakt van K-fold cross validation met shuffle en 5 folds.   

## Complexiteit verhogen

# Model training

## K-fold cross validation
Voor iedere fold is een model getraind met alle features en optimizer adam om het effect van de K-fold te laten zien ten opzichte van het baseline model van Milestone 1.

## Optimizer
De validation data heeft in veel van de gevallen een hoge loss. Daarom wordt er gezocht naar een betere optimizer voor het Neural Network. De eerste optimizer die in Milestone 1 is gebruikt is de 'adam' optimizer. Om te bepalen welke optimizer het beste past bij de data, zijn alle optimizers van Keras geprobeerd met verschillende learning rates. Om te bepalen welke optimizer met welke learning rate het beste bij het model past, wordt de RMSE loss opnieuw geplot.

# Model evaluation

## K-fold cross validation
Om het model te kunnen evalueren is de RMSE van alle 5 folds gemiddeld genomen. Dit is vervolgens geplot zoals eerder is gedaan in Milestone 1. Voor alle 5 folds zijn 300 epochs gebruikt. Het verschil in learning curves is te zien in de volgende plots. De eerste plot is de learning curve van het baseline model, de tweede plot is de learning curve van het model met K-fold cross validation.

< afbeelding 'Learning_curve_K_fold_all_features' >

Hieruit is te concluderen dat de spikes vooral te maken hebben gehad met niet representatieve validatie data, in plaats van met overfitting wat in de eerste instantie werd gedacht. De RMSE van dit model is nog steeds niet optimaal, dus wordt er verder gekeken naar het opimaliseren van het model doormiddel van het toevoegen van complexiteit. Zie *complexiteit verhogen*

## Difference plot
Wanneer een voorspeld percentage lager is dan het daadwerkelijke percentage positieve Covid tests, kan dit grotere gevolgen hebben dan wanneer het percentage hoger is voorspeld. Om dit in kaart te brengen is er een difference plot aan het model toegevoegd. Deze plot laat zien wat de verschillen zijn tussen de voorspelde percentages en de daadwerkelijke percentages van de validation data:

< afbeelding 'difference_plot'

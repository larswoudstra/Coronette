# Data analysis
Er is in eerste instantie gekeken naar de structuur van de data. Er zijn 94
features en 2700 samples in de training data. De test data heeft 893 samples.
De training data is opgesplitst in 70% training data en 30% validation data.
De data bestaat uit een one-hot encoding deel met 40 staten, 4 features over
ziektes die op corona lijken, 8 gedragsindicatoren, 5 indicatoren voor mentale
gezondheid en ten slotte het label: tested positive.

Om te beginnen worden alle 94 features gebruikt in het model, behalve de id.
In totaal zijn er dus 93 voorspellende features. Later kan aan de hand van model
analyse worden beoordeeld of enkele minder belangrijke features verwijderd
zouden kunnen worden. Wij verwachten dat de mentale gezondheidsindicatoren
minder sterke voorspellers zijn dan de andere features.

De data is gecontroleerd op NaN values, deze waren niet aanwezig. De labels zijn
gesplit van de rest van de data en in verschillende variabelen opgeslagen.
Alle data is numeriek: de 40 staten zijn one-hot encoded. Alle overige features
zijn percentages van groepen mensen in een bepaalde regio in een staat. De data
hoeft dus niet genormaliseerd te worden.

# Data pipeline
Om dit regressieprobleem op te lossen, wordt een Neural Network gecreëerd met
93 input nodes en 1 output nodes. We zijn begonnen met 1 hidden layer van 93
nodes. De output node geeft hier een voorspeld percentage.

# Model training  
Het Neural Network wordt getraind doormiddel van forward en backward propagation.
Doormiddel van gradient descent wordt het model geoptimaliseerd.

# Model Evaluation
Met RMSE wordt het gemiddelde verschil gegeven van het voorspelde percentage
met het daadwerkelijke percentage. Hoe lager dit verschil, hoe beter het model
het percentage kan voorspellen. Een plot is gemaakt om per sample dit verschil
te visualiseren. 

# Model 3.1: Keras weight initializers

## Data analysis

## Data pipeline

## Model Training

## Model evaluation

# Model 3.2: Dropout en Batch Normalization

## Data analysis

## Data pipeline

## Model Training

## Model evaluation

# Model 3.3: Verschillen analyseren
Om de verschillen tussen de voorspelde tested-positive waarden en de daadwerkelijke tested-positive waarden in kaart te brengen is er gebruik gemaakt van een histogram. Deze laat zien hoe vaak een bepaalde waarde van een verschil voorkomt. Idealiter zijn alle verschillen natuurlijk 0. Wanneer dit niet het geval is, geeft een normaalverdeling rondom 0 aan dat het model op een juiste manier wordt getraind. Wanneer er een andere verdeling te zien is, kan dit duiden op nog onbekende factoren die het model beïnvloeden.

## Data analysis
-

## Data pipeline
Om de verschillen te analyseren is gebruik gemaakt van het model met de 14 beste features met een architectuur van 14x5x1.

## Model Training
Het model is getraind volgens alle bovenstaande stappen die tot nog toe het beste de target values hebben kunnen voorspellen. Er is gebruik gemaakt van een batchsize 70 en 700 epochs. De initializer die gebruikt is is de He initializer.  

## Model evaluation
In de histogram die uit het bovenstaande model is voortgekomen is te zien dat de verschillen rondom 0 gecentreerd zijn. Dit betekent dat de meeste verschillen klein zijn. Slechts voor 3 van de ... samples wijkt het voorspelde percentage met meer dan 3% af van het daadwerkelijke percentage.

![Histogram verschillen](https://github.com/larswoudstra/Coronette/blob/main/docs/images/differences_hist.png)

# Model 3.4: Complexiteit verder verhogen

## Data analysis

## Data pipeline

## Model Training

## Model evaluation

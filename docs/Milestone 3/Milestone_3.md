# Model 3.1: Keras weight initializers

## Data analysis

## Data pipeline

## Model Training

## Model evaluation

# Model 3.2: Dropout en Batch Normalization
In een poging ons model nog verder te verbeteren, is er gekozen om te kijken naar
de effecten van een zeer complex netwerk, in combinatie met Dropout en BatchNormalization.
Uiteindelijk is uit de analyses gebleken dat het model overduidelijk beter werkt zonder
Dropout en BatchNormalization

## Data analysis
De relevante data analysis voor dit subhoofdstuk is terug te vinden aan het begin
van dit hoofdstuk. Er is gekozen om gebruik te maken van de 14 beste features, om
hiermee verschillende combinaties van layers/nodes te vergelijken. De data is gesplitst
in 80% trainingdata en 20% testdata. Bij het toetsen van de verschillende combinaties
is het model getraind met deze trainingdata en getest met de testdata. Hierdoor zijn
de resultaten goed vergelijkbaar, want alle modellen zijn met exact dezelfde data
getraind en getest.

## Data pipeline
Er zijn veel verschillende data pipelines gebruikt; elke variant had immers een andere
combinatie van dropout- en BatchNormalization layers.
Over het algemeen had het model 14 input nodes en minimaal één hidden layer met een
ReLU-activatie. Onder het kopje model evaluation is een tabel te zien met alle
combinatie van Dropout- en BatchNormalization layers.
Over het algemeen had het model 14 input nodes en minimaal één hidden layer met een
ReLU-activation. Onder het kopje model evaluation is een tabel te zien met alle
verschillende configuraties en bijbehorende resultaten. Er is ook gebruik gemaakt van
een zogenaamde He-initializer: deze zorgt ervoor dat de gewichten van het model
beter geïnitialiseerd worden, zodat het model minder vaak op een lokaal minimum
vast zou komen te zitten.

Vervolgens is voor elke hidden layer een Dropout-layer toegevoegd en na elke
hidden layer een BatchNormalization layer.

## Model Training
Het netwerk is getraind met 700 epochs en een batch size van 70, omdat uit eerdere
trainingen is gebleken dat deze combinatie tot de beste resultaten leidde. Het kan
natuurlijk het geval zijn dat dit anders is voor complexere modellen, maar door deze
waarden gelijk te houden kunnen de varianten beter vergeleken worden.
De MSE is gebruikt om het model te trainen.

## Model evaluation
RMSE is gebruikt om de uiteindelijke kwaliteit van het model te bepalen. Hoe lager
de RMSE, hoe beter. Er is gekeken naar de validatie-RMSE. Er is ook gekeken of het
model overfit door de training- en validatie-RMSE tegen elkaar te plotten. In
onderstaande tabel staan de geprobeerde configuraties met de bijbehorende RMSE.

![Tabel met configuraties](https://github.com/larswoudstra/Coronette/blob/main/docs/images/opgemaakte_tabel_batch%26drop.png)

In deze tabel is dus te zien dat er in alle verschillende vormen en mogelijkheden
geen verbetering is te zien, totdat er een erg kleine Dropout (0.1) wordt toegevoegd
op een simpel model met twee hidden layers (zie rij 7). De keren dat het model onder
de 1 komt is echter als de Dropout helemaal is weggehaald (zie rij 8 en 10).
BatchNormalization op zichzelf werkt ook niet goed, want dan wordt de RMSE 2.05.
In het vervolg is het dus aan te raden om het te houden op een simpeler model, want een
complexer model met Dropout en BatchNormalization leidt niet tot betere resultaten.

# Model 3.3: Verschillen analyseren
Om de verschillen tussen de voorspelde tested-positive waarden en de daadwerkelijke tested-positive waarden in kaart te brengen is er gebruik gemaakt van een histogram. Deze laat zien hoe vaak een bepaalde waarde van een verschil voorkomt. Idealiter zijn alle verschillen natuurlijk 0. Wanneer dit niet het geval is, geeft een normaalverdeling rondom 0 aan dat het model op een juiste manier wordt getraind. Wanneer er een andere verdeling te zien is, kan dit duiden op nog onbekende factoren die het model beïnvloeden.

## Data analysis
De data die wordt gebruikt is de data uit de 'covid.train.csv' file. Deze data is gesplitst in 80% trainingsdata (2160 samples) en 20% testdata (540 samples). Met de trainingsdata wordt het gehele model getraind. Aan de hand van de test data maakt het model voor elke sample een voorspelling van het aantal positieve covid-19 tests. Deze waarde wordt vergeleken met de al bekende waarde van het sample uit de 'covid.train.csv' file.

## Data pipeline
Om de verschillen te analyseren is gebruik gemaakt van het model met de 14 beste features met een architectuur van 14x5x1.

## Model Training
Het model is getraind volgens alle bovenstaande stappen die tot nog toe het beste de target values hebben kunnen voorspellen. Er is gebruik gemaakt van een batchsize van 70 en 700 epochs. De initializer die gebruikt is is de He initializer.  

## Model evaluation
In de histogram die uit het bovenstaande model is voortgekomen is te zien dat de verschillen rondom 0 gecentreerd zijn. Dit betekent dat de meeste verschillen klein zijn. Slechts voor 3 van de 540 samples wijkt het voorspelde percentage met meer dan 3% af van het daadwerkelijke percentage.

![Histogram verschillen](https://github.com/larswoudstra/Coronette/blob/main/docs/images/differences_hist.png)

# Model 3.4: Complexiteit verder verhogen

## Data analysis

## Data pipeline

## Model Training

## Model evaluation

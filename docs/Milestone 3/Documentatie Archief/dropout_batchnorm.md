# Model 3.2: Dropout en Batch Normalization
In een poging ons model nog verder te verbeteren, is er gekozen om te kijken naar
de effecten van een zeer complex netwerk, in combinatie met Dropout en Batchnormalization.
Uiteindelijk is uit de analyses gebleken dat het model overduidelijk beter werkt zonder
Dropout en Batchnormalization

## Data analysis
De relevante data analysis voor dit subhoofdstuk is terug te vinden aan het begin
van dit hoofdstuk. Er is gekozen om gebruik te maken van de 14 beste features, om
hiermee verschillende combinaties van layers/nodes te vergelijken. De data is gesplitst
in 80% trainingdata en 20% testdata. Bij het toetsen van de verschillende combinaties
is het model getraind met deze trainingdata en getest met de testdata. Hierdoor zijn
de resultaten goed vergelijkbaar, want alle modellen zijn met exact dezelfde data
getraind en getest.

## Data pipeline
Er zijn veel verschillende data pipelines gebruikt: elke variant had immers een andere
combinatie van dropout- en batchnormalization layers.
Over het algemeen had het model 14 input nodes en minimaal één hidden layer met een
relu-activation. Onder het kopje model evaluation is een tabel te zien met alle
verschillende configuraties en bijbehorende resultaten. Er is ook gebruik gemaakt van
een zogenaamde He-initializer: deze zorgt ervoor dat de gewichten van het model
beter geïnitialiseerd worden, zodat het model minder vaak op een lokaal minimum
vast zou komen te zitten.
Vervolgens is voor elke hidden layer een dropout-layer toegevoegd en na elke
hidden layer een batchnormalization layer.

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

![Tabel met configuraties](https://github.com/larswoudstra/Coronette/blob/main/docs/images/dropout_batchnorm_tabel.png)

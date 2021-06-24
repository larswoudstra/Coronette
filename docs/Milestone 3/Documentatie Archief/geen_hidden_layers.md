# Model 3.5: Geen hidden layers
Uit eerdere metingen en modellen is gebleken dat ons netwerk tot betere
resultaten leidt als deze simpeler is. Om deze reden wilden de onderzoekers nog
analyseren hoe goed een neuraal netwerk zonder hidden layers en activatiefunctie
zou presteren.

## Data analysis
Voor de data analysis kan gekeken worden naar eerdere hoofdstukken. Er is in dit
model getest met 14 (uit de feature selection) en 93 (alle) features. Er is
gebruikgemaakt van de 80% trainingdata voor het trainen en 20% testdata voor het
testen. Hierdoor kunnen eventuele verschillen in output niet veroorzaakt worden
door een andere validatieset.

## Data pipeline
Er zijn twee verschillende data pipelines geweest bij deze testen: 14 input nodes
en 93 input nodes. Bij beide netwerken is er geen gebruik gemaakt van
activatiefuncties, Dropout en BatchNormalization. Beide netwerken hadden 1
output node.

## Model training
Het model is getraind met de MSE voor gradient descent. Er is daarbij gebruikgemaakt
van 700 epochs en een batch size van 70, omdat uit eerdere analyses is gebleken dat
deze combinatie tot de beste resultaten leidt.

## Model evaluation
De daadwerkelijke prestaties van het model zijn gemeten met de RMSE. De waarden
hiervan zijn voor zowel de trainingdata als de testdata tegen elkaar geplot, zodat
er gekeken kan worden naar overfitting en underfitting. 

# Model 3.5: Geen hidden layers
Uit eerdere metingen en modellen is gebleken dat ons netwerk tot betere
resultaten leidt als deze simpeler is. Om deze reden wilden de onderzoekers nog
analyseren hoe goed een neuraal netwerk zonder hidden layers en activatiefunctie
zou presteren.

## Data analysis
Voor de data analysis kan gekeken worden naar eerdere hoofdstukken. Er is in dit
model getest met 14 (uit de feature selection) en 93 (alle) features. Er is
gebruikgemaakt van de 80% trainingdata voor het trainen, deze is opgesplitst in
5 delen met k-fold cross validation. Elk deel is een keer validatie geweest, en
het gemiddelde van de output daarvan is het resultaat.

## Data pipeline
Er zijn twee verschillende data pipelines geweest bij deze testen: 14 input nodes
en 93 input nodes. Bij beide netwerken is er geen gebruik gemaakt van
activatiefuncties, Dropout en BatchNormalization. Beide netwerken hadden 1
output node.

## Model training
Het model is getraind met de MSE voor gradient descent. Er is daarbij gebruikgemaakt
van 700 epochs en een batch size van 70, omdat uit eerdere analyses is gebleken dat
deze combinatie tot de beste resultaten leidt. Er is gebruik gemaakt van de
He-initializer, omdat deze uit eerdere analyses het best is gebleken voor onze data.
Diezelfde argumentatie geldt voor het gebruik van de Nadam optimizer.

## Model evaluation
De daadwerkelijke prestaties van het model zijn gemeten met de RMSE. De waarden
hiervan zijn voor zowel de trainingdata als de testdata tegen elkaar geplot, zodat
er gekeken kan worden naar overfitting en underfitting.

![Tabel met resultaten]()

Er zijn voor de onderzoekers verrassende resultaten uit de tests gekomen. Zo lijkt
93 input nodes het over het algemeen, op een paar uitschieters na, het beter te doen
dan 14 input nodes. Bovendien geven sommige configuraties met 93 input nodes het
beste resultaat tot nu toe (zie rij 2 en rij 8).
De resultaten van de modellen met 14 input nodes zijn redelijk te vergelijken met
de resultaten van modellen met 14 input nodes en 1 hidden layer, deze gaven namelijk
meestal een output van ongeveer 0.91.
In conclusie kan er in het vervolg beter gewerkt worden met een model met 93
input nodes, geen hidden layers, 3000 of 10000 batch size en 70 of 1000 epochs.

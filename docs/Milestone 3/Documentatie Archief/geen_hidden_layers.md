# Model 3.4: Geen hidden layers
Uit eerdere metingen en modellen is gebleken dat ons netwerk tot betere resultaten leidt als deze simpeler is. Om deze reden wilden de onderzoekers nog analyseren hoe goed een neuraal netwerk zonder hidden layers en activatiefunctie zou presteren.

## Data analysis
Voor de data analysis kan gekeken worden naar eerdere hoofdstukken. Er is in dit model getest met 14 (uit de feature selection) en 93 (alle) features. Er is gebruikgemaakt van de 80% trainingdata voor het trainen, deze is opgesplitst in 5 delen met k-fold cross validation. Elk deel is een keer validatie geweest, en het gemiddelde van de output daarvan is het resultaat.

## Data pipeline
Er zijn twee verschillende data pipelines geweest bij deze testen: 14 input nodes en 93 input nodes. Bij beide netwerken is er geen gebruik gemaakt van activatiefuncties, Dropout en BatchNormalization. Beide netwerken hadden 1 output node.

## Model training
Het model is getraind met de MSE voor gradient descent. Er is daarbij gebruikgemaakt van 700 epochs en een batch size van 70, omdat uit eerdere analyses is gebleken dat deze combinatie tot de beste resultaten leidt. Er is gebruik gemaakt van de He-initializer, omdat deze uit eerdere analyses het best is gebleken voor onze data. Diezelfde argumentatie geldt voor het gebruik van de Nadam optimizer.

## Model evaluation
De daadwerkelijke prestaties van het model zijn gemeten met de RMSE. De waarden hiervan zijn voor zowel de trainingdata als de testdata tegen elkaar geplot, zodat er gekeken kan worden naar overfitting en underfitting.

![Tabel met resultaten](https://github.com/larswoudstra/Coronette/blob/main/docs/images/opgemaakte_tabel_zonderhiddenlayers.png)

Er zijn geen hele verrassende resultaten uit de analyses gekomen. Geen enkele configuratie deed het beter dan het beste resultaat tot nu toe (0.92). Een interessante waarneming is wel dat de RMSE met 14 input nodes bijna consistent net wat lager is dan de RMSE met 93 input nodes. Het is dus waarschijnlijk een goede keuze geweest van de onderzoekers om verder te gaan met de 14 beste features.
In het vervolg is het geen slimme keuze om gebruik te maken van geen hidden layers in het neurale netwerk, want dit leidt niet tot betere resultaten.

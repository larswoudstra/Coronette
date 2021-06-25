# Machine Learning project: Algoritme voor de voorspelling van het aantal COVID gevallen voor een dag
### Door: Merel, Nina, Kelly en Lars

In dit project is er onderzoek gedaan naar een algoritme dat de dagelijkse COVID gevallen kan voorspellen.

# Data analysis
De data die voor dit onderzoek gebruikt wordt, is afkomstig van [Kaggle](https://www.kaggle.com/c/ml2021spring-hw1/data). Met deze data wordt het percentage nieuwe positieve Covid-19 testuitslagen getracht te voorspellen in 40 verschillende staten in de Verenigde Staten. Per staat zijn ongeveer 66 steekproeven gedaan, waarbij iedere steekproef van een groep mensen uit een staat dient als een gemiddelde schatting voor de gehele staat. In de data komt een sample dus overeen met een steekproef.

Zowel de training- als de testdata zijn .csv-bestanden, welke individueel van elkaar worden ingeladen als Pandas DataFrames. De trainingdata bestaat uit 2700 samples, de test data uit 893 samples. De 94 features die in deze datasets worden meegenomen betroffen onder andere een ID, de 40 staten (*one-hot encoded*) en 3 sets van 17 variabelen, waarbij de sets 3 opeenvolgende dagen vertegenwoordigen. Ieder van deze 17 variabelen zijn dus over een periode van 3 dagen gemeten. Deze 17 variabelen betreffen 4 features over ziektes die op corona lijken, 8 gedragsindicatoren en 5 indicatoren voor mentale gezondheid. Ten slotte bevatten de datasets 2 variabelen die het percentage positieve testuitslagen op dag 1 en dag 2 vertegenwoordigen. Met behulp van deze 94 features wordt het percentage nieuwe positieve testuitslagen op dag 3 getracht te voorspellen. Dit percentage is dan ook de *target value* van dit onderzoek. De variabelen van de data zijn te zien in onderstaande afbeelding.  

![Data explanation](https://github.com/larswoudstra/Coronette/blob/main/docs/images/data_explanation.jpg)

De kolom met ID’s wordt verwijderd, aangezien deze alleen als index dient. De verdere analyses worden vooralsnog uitgevoerd met de resterende 93 features. Later kan er aan de hand van het model worden beoordeeld of enkele minder relevante features verwijderd moeten worden.

In zowel de training- als testdata worden de features en de target value van elkaar gesplitst. De trainingdata wordt opgesplitst in 70% trainingdata en 30% validatiedata, wat resulteerde in 1890 trainingsamples en 810 validatiesamples. Aangezien alle data numeriek is, hoeft de data niet genormaliseerd te worden en kan er een regressie uitgevoerd worden om de target value voor dag 3 te voorspellen.

# Data pipeline
Om dit regressieprobleem op te lossen, wordt een Neuraal Netwerk (NN) gebruikt. Deze zal bestaan uit 93 input nodes, één voor iedere feature, en 1 output node om de uiteindelijke voorspelling – het percentage nieuwe positieve Covid-19-testuitslagen – te maken. De input is dus 1890x93-matrix. Voor de eerste versie van het model is gekozen voor een enkele hidden layer met 93 nodes. De hidden layer heeft dus dezelfde dimensies als de input layer. Voor de hidden layer is een *ReLU*-activatiefunctie gebruikt, aangezien hiermee een non-lineair en complex probleem opgelost kan worden. Voor de output layer wordt geen activatiefunctie gebruikt, omdat er hier een numerieke voorspelling wordt gedaan, waarbij de data niet getransformeerd dient te worden met een activatiefunctie. De output layer heeft dimensies 1890x1.

# Model training
Het NN wordt getraind doormiddel van forward en backward propagation. Met full-batch gradient descent wordt het model geoptimaliseerd. De *Mean Squared Error* (MSE) wordt gebruikt om het model te trainen, zodat grote verschillen zwaarder meewegen bij het trainen. De daadwerkelijke kwaliteit van het model wordt gemeten met de *Root Mean Squared Error* (RMSE), aangezien hiermee de error beter te interpreteren is. Een RMSE van 2.0 zou bijvoorbeeld ook duiden op een gemiddeld verschil van 2.0 tussen het voorspelde percentage en het daadwerkelijke percentage. Er wordt gebruikgemaakt van de *Adam* optimizer, omdat dit de default optimizer is. In het vervolg worden echter ook andere mogelijke optimizers overwogen. Bij de Adam optimizer hoort een standaard *learning rate* van 0.01. Er is gewerkt met 500 epochs, welke genoeg lijkt om de learning curve correct weer te geven om het model te kunnen evalueren.

# Model Evaluation
Met RMSE wordt het gemiddelde verschil gegeven van het voorspelde percentage met het daadwerkelijke percentage. Hoe lager dit verschil, hoe beter het model het percentage kan voorspellen. Bij dit initiële model ligt dit getal rond 1.13. Dit betekent dat de gemiddelde afwijking van het voorspelde percentage met het actuele percentage slechts 1.13% is. Voor een eenvoudig model als dit, is dit al een hele goede prestatie. Dit wijst er op dat het op te lossen probleem simpeler lijkt dan in de eerste instantie werd gedacht. In het vervolg streven we ernaar om de RMSE zo laag mogelijk onder de 1 te krijgen.

In eerste instantie worden de output van de validatiedata en de output van de daadwerkelijke data in eenzelfde figuur geplot, zodat gekeken kan worden of deze lijnen enigszins overeenkomen. Om de figuur beter interpreteerbaar te maken, wordt de ene lijn transparanter gemaakt dan de ander. Hier is te zien dat de lijnen erg goed overeenkomen. Desondanks wordt de grafiek als erg onduidelijk ervaren, omdat hierin eerder de overeenkomsten dan de verschillen benadrukt worden.

![real_vs_predicted_plot](https://github.com/larswoudstra/Coronette/blob/main/docs/images/predicted_vs_real_plot_ml1.png)

Vandaar dat er een andere grafiek wordt gemaakt waarin de verschillen tussen de voorspelde data en de daadwerkelijke data geplot worden. Hierin is te zien dat de meeste voorspellingen niet meer dan 2.0% afwijken van de daadwerkelijke data, hoewel er ook een aantal uitschieters te zien die maximaal een verschil van 4.0% laten zien. Deze voorspelling moet en kan beter. Dit hopen we te bewerkstelligen door het model complexer te maken.

![difference_plot](https://github.com/larswoudstra/Coronette/blob/main/docs/images/difference_plot_ml1.png)

Tenslotte wordt er een plot gemaakt waarin de trainingkosten en validatiekosten geplot worden. Er zijn in deze grafiek grote pieken te zien, die steeds minder heftig worden. We verwachten dat dit deels veroorzaakt wordt door overfitting en een gebrek aan normalisatie. Dit zullen we in een vervolgstap dus verwerken in het model. Hierna zal er ook gekeken worden naar de effecten van verschillende aantallen epochs. Ook zouden de pieken in de grafiek veroorzaakt kunnen worden doordat de validatiedata niet representatief is voor de populatie. In een volgende versie wordt geprobeerd dit op te lossen door middel van k-fold cross validation.

![training and validation losses](https://github.com/larswoudstra/Coronette/blob/main/docs/images/losses_plus_mental_health.png)

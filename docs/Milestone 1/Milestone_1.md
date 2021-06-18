# Data analysis
De data die voor dit onderzoek gebruikt wordt, is afkomstig van [Kaggle](https://www.kaggle.com/c/ml2021spring-hw1/data). Met deze data wordt het percentage nieuwe positieve Covid-19 testuitslagen getracht te voorspellen in 40 verschillende staten in de Verenigde Staten. Per staat zijn ongeveer 66 steekproeven gedaan, waarbij iedere steekproef van een groep mensen uit een staat dient als een gemiddelde schatting voor de gehele staat. Hierbij komt een sample dus overeen met een steekproef.

Zowel de training- als de testdata zijn .csv-bestanden, welke individueel van elkaar worden ingeladen als Pandas DataFrames. De trainingdata bestaat uit 2700 samples, de test data uit 893 samples. De 94 features die in deze datasets worden meegenomen betroffen onder andere een ID, de 40 staten (*one-hot encoded*) en 3 sets van 17 variabelen, waarbij de sets 3 opeenvolgende dagen vertegenwoordigen. Ieder van deze 17 variabelen zijn dus over een periode van 3 dagen gemeten. Deze 17 variabelen betreffen 4 features over ziektes die op corona lijken, 8 gedragsindicatoren en 5 indicatoren voor mentale gezondheid. Ten slotte bevatten de datasets 2 variabelen die het percentage positieve testuitslagen op dag 1 en dag 2 vertegenwoordigen. Met behulp van deze 94 features wordt het percentage nieuwe positieve testuitslagen op dag 3 getracht te voorspellen. Dit percentage is dan ook de *target value* van dit onderzoek.

Eventuele *missing values* worden verwijderd uit de trainingdata, evenals de kolom met ID’s, aangezien deze alleen als index dient. De verdere analyses worden vooralsnog uitgevoerd met de resterende 93 features. Later kan er aan de hand van het model worden beoordeeld of enkele minder relevante features verwijderd moeten worden.

<!-- Zo wordt er bijvoorbeeld op voorhand al verwacht dat de mentale-gezondheids-indicatoren minder sterke voorspellers zullen zijn dan de andere features.  -->

In zowel de training- als testdata worden de features en de target value van elkaar gesplitst. De trainingdata wordt opgesplitst in 70% trainingdata en 30% validatiedata. Aangezien alle data numeriek is, hoeft de data niet genormaliseerd te worden en kan er een regressie uitgevoerd worden om de voorspelling voor dag 3 te doen.

# Data pipeline
Om dit regressieprobleem op te lossen, wordt een Neuraal Netwerk (NN) gebruikt. Deze zal bestaan uit 93 input nodes, één voor iedere feature, en 1 output node om de uiteindelijke voorspelling – het percentage nieuwe positieve Covid-19-testuitslagen – te maken. Voor de eerste versie van het model is gekozen voor een enkele hidden layer met 93 nodes. Voor de hidden layer is een *ReLU*-activatiefunctie gebruikt, aangezien hiermee een non-lineair en complex probleem opgelost kan worden. Voor de output layer wordt geen activatiefunctie gebruikt, omdat er hier een numerieke voorspelling wordt gedaan, waarbij de data niet getransformeerd dient te worden met een activatiefunctie.
**Nog iets zeggen over dimensies van de layers/nodes?**

# Model training
Het NN wordt getraind door middel van forward en backward propagation. Met full-batch gradient descent wordt het model geoptimaliseerd. De *Mean Squared Error* (MSE) wordt gebruikt om het model te trainen. De daadwerkelijke kwaliteit van het model wordt gemeten met de *Root Mean Squared Error* (RMSE). Er wordt gebruikgemaakt van de *Adam* optimizer, omdat deze bekend is bij de onderzoekers. In het vervolg worden echter ook andere mogelijke optimizers overwogen. Bij de Adam optimizer hoort een standaard *learning rate* van 0.01. Er zijn verschillende hoeveelheden epochs uitgeprobeerd, maar voor nu wordt er met vijfhonderd epochs gewerkt.

# Model Evaluation
Met RMSE wordt het gemiddelde verschil gegeven van het voorspelde percentage met het daadwerkelijke percentage. Hoe lager dit verschil, hoe beter het model het percentage kan voorspellen. Bij dit initiële model ligt dit getal rond 1.13. In het vervolg streven we ernaar om dit getal zo laag mogelijk onder de 1 te krijgen. Er wordt een plot gemaakt om per sample dit verschil te visualiseren. Zie figuur X in de images. Er zijn in deze grafiek grote pieken te zien, die steeds minder heftig worden. We verwachten dat dit deels veroorzaakt wordt door overfitting en een gebrek aan normalisatie. Dit zullen we in een vervolgstap dus verwerken in het model. Er wordt ook een andere grafiek geplot, zie figuur Y, waarin de output van de validatiedata vergeleken wordt met de output van de trainingdata. Er blijken grote verschillen te zijn. **???** De grafiek is echter nog niet perfect, omdat de lijnen over elkaar heen liggen, waardoor de verschillen niet goed te zien zijn. In het vervolg gaan we het verschil plotten, en dit proberen te minimaliseren.

Aangezien het model nu nog op sommige punten lijkt te overfitten, wordt er eerst aan dit probleem gewerkt, voordat er verder naar het aantal epochs wordt gekeken.

**Misschien nog iets zeggen over dat we misschien te weinig validatiedata hebben en dat we dit gaan proberen op te lossen d.m.v. cross validation?**

**We moeten die images nog ff toevoegen**

# Model 3.1: Keras weight initializers
Tot nu toe is er gebruikt gemaakt van Keras' default Glorot Uniform (ook wel 'Xavier') weight initializer (zie ook Keras Dense Layer documentatie: https://keras.io/api/layers/core_layers/dense/).

## Data analysis
Tijdens het valideren van ons huidige model die deze initializer gebruikt is het opgevallen dat de RMSE soms op een erg hoge value start (vaak hoger dan 20). Daarnaast wordt in de meeste gevallen  de error geminimaliseerd tot een waarde rond de 1, maar in andere gevallen kan het voorkomen dat de weights zodanig zijn geinitialiseerd dat er meer kans is dat het algoritme vast komt te zitten op zogenaamde 'saddle points'. Dit leidt vervolgens weer tot hoge RMSE waarden bij het convergeren, zoals te zien in de afbeelding hieronder. De RMSE waarde is 7.66.

![Afbeelding met het probleem van de default initializer](https://github.com/larswoudstra/Coronette/blob/main/docs/images/Default_initializer_problem.png)
*Let op de schaal van de figuur*

## Data pipeline
Om dit probleem op te lossen zijn er andere initializers voorgesteld in dit project. Keras' default weight initializer werkt het beste op symmetrische activatie functies, zoals tanh en sigmoid (voor referentie, zie: https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79). In het huidige neurale netwerk worden er ReLU activatiefuncties gebruikt, deze zijn in tegenstelling tot de hiervoor genoemde activatiefuncties niet symmetrisch. Om deze reden hebben He et. al (2005) onderzoek gedaan naar een geschikte weight initializator voor dergelijke asymmetrische non-lineaire activatiefuncties. 'He' initialization werkt met getallen die uit een normaalverdeling komen en vermenigvuldigd worden met *sqrt(2)/sqrt(n)*, waarbij *n* het aantal inputs van de vorige layer voorstelt (in ons geval dus de 'output nodes', omdat we werken met een fully connected neuraal netwerk). He initialisatie geeft dan uiteindelijk getallen met een variantie van 1 voor deze type activatiefuncties.

## Model training
Het netwerk is getraind met drie verschillende initializers: de default Glorot Uniform (Xavier) initializer, de He initializer en de RandomNormal initializer. De laatste initializer genereert getallen uit een normaalverdeling zoals de He initializer dat doet, maar vermenigvuldigd dit niet met een bepaalde waarde die gerelateerd is aan het aantal inputs van de vorige layer. Voor het trainen zijn er 400 epochs en een batch size van 40 gebruikt omdat we voornamelijk geinteresseerd zijn in de beginwaarde van de RMSE na de initialisering en kan er al na een paar honderd epochs aan de hand van het verloop van de RMSE bepaald worden of de RMSE op een waarde rond de 1 zal zitten. Verder komt het model overeen met het huidig gebruikte model met een architectuur van 14, 5, 1 en een Nadam optimizer. Er is gevalideerd over 5 folds en hiervan is de gemiddelde RMSE waarde genomen.

## Model evaluation
In de afbeelding hieronder is het verschil te zien tussen de verschillende initializers (He, Glorot Uniform (Xavier) en RandomNormal).

![Afbeelding van de verschillende initializers](https://github.com/larswoudstra/Coronette/blob/main/docs/images/initializers2_final.png)

**nog duidelijk maken dat we met een andere waarde voor batch size en epochs werken omdat dat sneller is en dat om deze reden de exacte RMSE die we zouden kunnen krijgen niet klopt, maar je kan hier wel goed het verschil zien tussen de initializers**

Zoals verwacht start de He initializator op een lagere RMSE waarde dan de andere initializers. Ook is te zien dat de He initializer op een lagere RMSE waarde uitkomt ten opzichte van de Glorot initializer. Opvallend is hier wel dat de RandomNormal initializer ongeveer op dezelfde waarde uitkomt als bij het gebruik van de He initializer. Dit is waarschijnlijk te danken aan de vergelijkbare manier waarop de twee initializers werken, namelijk door een getal te kiezen uit de normaalverdeling *(vandaar: RandomNormal)*. Omdat de RandomNormal deze getallen niet vermenigvuldigt met een term zoals in de He initializer, kan dit de reden zijn waarom de RMSE waarde in het begin ook hoger is voor de RandomNormal initializer.

Als de 'He' initializer wordt toegepast op de configuratie zoals deze is gebruikt om de afbeelding in de sectie 'Data pipeline' te maken, dan is in de afbeelding hieronder te zien dat deze initializer een lagere RMSE waarde heeft (0.92). Dit blijft ook het geval wanneer je deze opzet meerdere keren laat runnen.

![Afbeelding van de oplossing m.b.v. de He initializer](https://github.com/larswoudstra/Coronette/blob/main/docs/images/He_initializer_solution.png)

Vanaf dit punt zullen we dan ook verder werken met de He initializer.

# Model 3.2: Dropout en Batch Normalization
In een poging ons model nog verder te verbeteren, is er gekozen om te kijken naar de effecten van een zeer complex netwerk, in combinatie met Dropout en BatchNormalization. Uiteindelijk is uit de analyses gebleken dat het model overduidelijk beter werkt zonder Dropout en BatchNormalization

## Data analysis
De relevante data analysis voor dit subhoofdstuk is terug te vinden aan het begin van dit hoofdstuk. Er is gekozen om gebruik te maken van de 14 beste features, om hiermee verschillende combinaties van layers/nodes te vergelijken. De data is gesplitst in 80% trainingdata en 20% testdata. Bij het toetsen van de verschillende combinaties is het model getraind met deze trainingdata en getest met de testdata. Hierdoor zijn de resultaten goed vergelijkbaar, want alle modellen zijn met exact dezelfde data getraind en getest.

## Data pipeline
Er zijn veel verschillende data pipelines gebruikt; elke variant had immers een andere combinatie van Dropout- en BatchNormalization layers. Over het algemeen had het model 14 input nodes en minimaal één hidden layer met een ReLU-activatie. Onder het kopje model evaluation is een tabel te zien met alle combinatie van Dropout- en BatchNormalization layers. Over het algemeen had het model 14 input nodes en minimaal één hidden layer met een ReLU-activation. Onder het kopje model evaluation is een tabel te zien met alle verschillende configuraties en bijbehorende resultaten. Er is ook gebruik gemaakt van een zogenaamde He-initializer: deze zorgt ervoor dat de gewichten van het model beter geïnitialiseerd worden, zodat het model minder vaak op een lokaal minimum vast zou komen te zitten.

Vervolgens is voor elke hidden layer een Dropout-layer toegevoegd en na elke hidden layer een BatchNormalization layer.

## Model Training
Het netwerk is getraind met 700 epochs en een batch size van 70, omdat uit eerdere trainingen is gebleken dat deze combinatie tot de beste resultaten leidde. Het kan natuurlijk het geval zijn dat dit anders is voor complexere modellen, maar door deze waarden gelijk te houden kunnen de varianten beter vergeleken worden. De MSE is gebruikt om het model te trainen.

## Model evaluation
RMSE is gebruikt om de uiteindelijke kwaliteit van het model te bepalen. Hoe lager de RMSE, hoe beter. Er is gekeken naar de validatie-RMSE. Er is ook gekeken of het model overfit door de training- en validatie-RMSE tegen elkaar te plotten. In onderstaande tabel staan de geprobeerde configuraties met de bijbehorende RMSE.

![Tabel met configuraties](https://github.com/larswoudstra/Coronette/blob/main/docs/images/Final_tabel_dropout.png)

In de tabel is  te zien dat er met grote hidden layers, Dropout en BatchNormalization een aantal keren overfitting optrad. Dit was af te lezen in de bijbehorende plot, want de validatiekosten gingen omhoog, terwijl de trainingkosten verder naar beneden gingen.

![Tabel met overfitting](https://github.com/larswoudstra/Coronette/blob/main/docs/images/2drop%26batch.png)

In de tabel is ook te zien dat er in alle verschillende vormen en mogelijkheden geen verbetering is te zien, totdat er een erg kleine Dropout (0.1) wordt toegevoegd op een simpel model met twee hidden layers (zie rij 7). De keren dat het model onder de 1 komt is echter als de Dropout helemaal is weggehaald (zie rij 8 en 10). BatchNormalization op zichzelf werkt ook niet goed, want dan wordt de RMSE 2.05.

In het vervolg is het dus aan te raden om het te houden op een simpeler model, want een complexer model met Dropout en BatchNormalization leidt niet tot betere resultaten.

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

![Tabel met resultaten](https://github.com/larswoudstra/Coronette/blob/main/docs/images/Final_tabel_geenhiddenlayers.png)

Er zijn geen hele verrassende resultaten uit de analyses gekomen. Geen enkele configuratie deed het beter dan het beste resultaat tot nu toe (0.92). Een interessante waarneming is wel dat de RMSE met 14 input nodes bijna consistent net wat lager is dan de RMSE met 93 input nodes. Het is dus waarschijnlijk een goede keuze geweest van de onderzoekers om verder te gaan met de 14 beste features.
In het vervolg is het geen slimme keuze om gebruik te maken van geen hidden layers in het neurale netwerk, want dit leidt niet tot betere resultaten.

# Model 3.5: Learning Rates

Ten slotte is er gekeken of het kiezen van verschillende learning rates in optimizer Nadam nog invloed heeft op de RMSE.

## Data analysis
Zoals in Milestone 2 is beschreven blijft de validation RMSE vaak hangen op een waarde rond de 0.94 en 1. Dit zou kunnen komen door zogenaamde zadelpunten. Als dit het geval is, zou middels het aanpassen van de learning rate de RMSE validatie waarde nog lager kunnen worden.

## Data pipeline
Om te kijken of er sprake is van eventuele zadelpunten waardoor we niet veel lager komen dan onze huidige RMSE, wordt er getuned met verschillende learning rates voor de Nadam optimizer. Eerst wordt de default learning rate toegepast van 0.001, vervolgens 0.003, daarna 0.01 en als laatste een learning rate van 0.03.

## Model Training
Voor het trainen met de verschillende learning rates zijn er iedere keer 700 epochs gebruikt met een batch size van 70. Verder komt het model overeen met het huidig gebruikte model met een architectuur van 14, 5, 1, een Nadam optimizer en de He initializer. Er is gevalideerd over 5 folds en hiervan is de gemiddelde RMSE waarde genomen.

## Model evaluation

![Tabel met resultaten](https://github.com/larswoudstra/Coronette/blob/main/docs/images/Final_tabel_geenhiddenlayers.png)

# Conclusie
Na het runnen van het model met alle bovenstaande configuraties met de test data is er een RMSE van 0.92 bereikt. Dit betekent dat de gemiddelde afwijking van het voorspelde percentage positieve covid-19 tests 0.92% is. Al met al kan er geconcludeerd worden dat dit een ontzettend accuraat model is om het aantal positieve covid-19 tests te kunnen voorspellen. Het model gaf de volgende learning curve:

![Learning curve test data](Url)

Daarnaast zijn ook hier de verschillen gecentreerd rondom 0, wat aangeeft dat slechts enkele gevallen een veel grotere afwijking dan het gemiddelde hebben:

![Test data histogram](Url)

## Discussie
Op de laatste dag van het onderzoek werd geconcludeerd dat sommige aanpassingen in het model niet met de k-fold cross validation functie zijn geëvalueerd, maar met de test data in de test NN functie. Hierdoor zijn enkele RMSE validatie waarden lager uitgevallen dan het eigenlijk hoort te zijn. Het model testen met de test data in de testNN functie viel namelijk in alle gevallen lager uit dan wanneer het model met 5 K-folds werd gebruikt. De verschillen hiertussen zijn gemiddeld 0.08%, maar toch niet verwaarloosbaar. In een volgend onderzoek kan uitgezocht worden wat het effect is van het gebruik van enkel en alleen k-fold cross validation om het model te evalueren.

Daarnaast kan er onderzocht worden waarom de test data gemiddeld een 0.08% lagere afwijking heeft dan de validatie afwijking met k-fold cross validation. In onderstaande tabel zijn de gemiddelde waarden van alle 14 features per k-fold en de gemiddelde waarden van alle 14 features van de test-data te zien. Deze tabel laat zien dat ook de gemiddelde verschillen tussen de validatiedata en testdata erg klein zijn. 

![Tabel verschillen](Url)

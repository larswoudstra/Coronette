# Model 2.1: RMSE Metric

## Model Training
Voor het trainen van het model wordt de MeanSquaredError gebruikt.

## Model evaluation
Om te bepalen hoe goed het model nu daadwerkelijk is, wordt er gebruik gemaakt van een RootMeanSquaredError learning curve. Deze plot geeft het gemiddelde verschil aan tussen de daadwerkelijke waarden en de voorspelde waarden; bij een hoge RMSE is het verschil groot, bij een kleine RMSE is het verschil klein.

![Baseline model met 93 features en Nadam](https://github.com/larswoudstra/Coronette/blob/main/docs/images/baselinemodel_nadam_k93.png)

# Model 2.2: Optimizers
## Data analysis
--

## Data pipeline
De validatiedata heeft in veel van de gevallen een hoge loss. Daarom wordt er gezocht naar een betere optimizer voor het Neural Network. De eerste optimizer die in Milestone 1 is gebruikt is de 'adam' optimizer. Om te bepalen welke optimizer het beste past bij de data, zijn alle optimizers van Keras geprobeerd met verschillende learning rates. Om te bepalen welke optimizer met welke learning rate het beste bij het model past, wordt de RMSE loss opnieuw geplot.

## Model training
Om een eerste indruk te krijgen van de verschillende optimizers en deze goed te kunnen vergelijken wordt er in eerste instantie gewerkt met 100 epochs. Een analyse van iedere mogelijke optimizer met 100 epochs en ieders default learning rate - 0.01 voor SGD en  0.001 voor de overige optimizers - wijst een aantal interessante dingen uit. Zo blijken SGD (Validatie RMSE = 7.81), Adam (Validatie RMSE = 1.27), Adadelta (Validatie RMSE = 3.26) en Adagrad (Validatie RMSE = 2.20) niet de beste keuzes te zijn, omdat deze modellen al vrij snel in het trainingproces (bijvoorbeeld na 20 epochs) al niet verder lijken te leren met nog relatief hoge kosten.

## Model evaluation
Adamax (Validatie RMSE = 1.08), Nadam (Validatie RMSE = 1.07), RMSProp (Validatie RMSE = 1.13) en Ftrl (Validatie RMSE = 1.24) blijken na de analyse met 100 epochs daarentegen nog wel optimizers die mogelijk het overwegen waard zijn, omdat deze de laagste validatiekosten opleveren en/of nog duidelijk aan het leren zijn tegen het einde van het trainen.

Na een analyse met 300 epochs en nog altijd ieders default learning rate lijkt Nadam (Validatie RMSE = 0.94) de beste keuze te zijn voor het regressieprobleem. Bij RMSProp stegen de validatiekosten namelijk naarmate het aantal epochs toenam (Validatie RMSE = 1.37). Ftrl (Validatie RMSE = 1.23) leek niet veel meer te leren met 200 extra epochs, maar de kostenplot was wel erg glad. Hoewel Adamax het ook zeker niet slecht doet (Validatie RMSE = 0.98), lijkt dit model al gestopt te zijn met leren waar Nadams validatiekosten nog dalen. Bovendien resulteerde het model met de Nadam optimizer in de laagste kosten, en deze lijken zelfs nog een beetje te dalen na 300 epochs.

Al met al lijkt de Nadam optimizer de beste keuze, maar Adamax en Ftrl moeten in het achterhoofd gehouden worden, aangezien deze mogelijk betere resultaten geven met andere learning rates, extra hidden layers of extra hidden nodes.

Er wordt dus nog steeds full batch gradient descent gebruikt, maar dan nu met de 'Nadam'-optimizer. Er worden 300 epochs gebruikt, omdat dit aantal voldoende informatie lijkt te geven over de werking van het model.

# Model 2.3: K-fold cross validation

## Data analysis
--

## Data Pipeline
In Milestone 1 is er gebruikgemaakt van de train_test_split-functie van sklearn om de data te verdelen in 70% trainingdata en 30% validatiedata.

Er wordt verwacht dat de pieken in de grafiek veroorzaakt worden door een te kleine validatieset, die bovendien niet representatief is voor de data waarvoor het model getraind is. Om dit op te lossen is er gebruik gemaakt van K-fold cross validation met shuffle en 5 folds.  

Er is tevens nagedacht om de K-folds te stratificeren met StratifiedKfold zodat er een goede ratio is van samples tussen de staten. Dit zou kunnen zorgen voor validatiedata met nog een hogere representativiteit. Echter bleek na het uitproberen van de StratifiedKfold en het lezen van de documentatie dat de targets (‘tested_positive’ in dit geval) categorische waarden moeten aannemen, wat hier dus niet het geval is. Om deze reden is er besloten om toch te werken met een klassieke K-fold cross validation. Het risico blijft dat de validatiedata niet representatief zou kunnen zijn door bijvoorbeeld samples die enkel uit twee staten komen, echter is dit in de praktijk tot dusver niet voorgekomen.

## Model training
Voor iedere fold is een model getraind met alle features (93) en optimizer Nadam om het effect van de K-fold te laten zien ten opzichte van het baseline model van Milestone 1. Zowel de modellen van de K-fold als het baseline model had de architectuur 93x93x1.

## Model evaluation
Om het model te kunnen evalueren is de RMSE van alle 5 folds gemiddeld genomen. Dit is vervolgens geplot zoals eerder is gedaan in Milestone 1. Voor alle 5 folds zijn 300 epochs gebruikt. Het verschil in learning curves is te zien in de volgende plots. De eerste plot is de learning curve van het baseline model, de tweede plot is de learning curve van het model met K-fold cross validation.

![Baseline model met 93 features en Nadam](https://github.com/larswoudstra/Coronette/blob/main/docs/images/baselinemodel_nadam_k93.png)
![Model na k-fold met 93 features en Nadam](https://github.com/larswoudstra/Coronette/blob/main/docs/images/k_fold_k93_1hidden.png)

Aangezien er minder spikes te zien zijn, kan er waarschijnlijk geconcludeerd worden dat deze inderdaad veroorzaakt werden door niet-representatieve validatiedata. Dit lijkt dus opgelost te worden met K-fold cross validation. Daarom wordt deze methode in het vervolg ook gebruikt om het model te evalueren.  

# Model 2.4: Feature selection

## Data analysis
Door middel van feature selection is er gekeken naar welke features de uiteindelijke target value het beste kunnen voorspellen.

## Data pipeline
Voor de feature selection is gebruik gemaakt van SelectKBest-analyse. Deze analyseert de correlatie tussen alle verschillende features en de target feature en gebruikt hiervoor de scorefunctie 'f_regression'. Features met hogere correlatiescores hebben dus een sterker verband met de target feature en zijn daarmee relevanter om mee te nemen in het model.

Dergelijke *feature selection* helpt niet alleen overfitting te verminderen, het zou ook de kosten en trainingstijd moeten terugdringen. Hiervoor moet worden onderzocht welke variabelen de beste voorspellers zijn voor de target feature (het percentage positieve testuitslagen). Bij sklearn's SelectKBest-functie moet een k, ofwel het aantal features met de hoogste scores, gedefinieerd worden.

## Model training
De SelectKBest-functie wordt uitgevoerd voor alle 93 features in de data. Vervolgens wordt aan de hand van de score die elke feature heeft gekregen, bepaald welke features het beste de target value kunnen voorspellen. Aan de hand van deze score wordt dan gekeken welke features belangrijk zijn om te behouden en welke eventueel verwijderd zouden kunnen worden.

## Model evaluation
De correlatiescore van alle 93 features (k="all") is weergeven in onderstaande barplot.

![bar plot](https://github.com/larswoudstra/Coronette/blob/main/docs/images/best_features_barplot.png)

De hoeveelheid features met de hoogste pieken geeft een idee van de benodigde waarde van k. Hieruit blijken 14 features met de hoogste pieken, en dus de hoogste correlatiescore. Deze worden dus sowieso meegenomen. Na de evaluatie wordt er gekeken welke hoeveelheid features daadwerkelijk tot het beste resultaat leidt.

![best 14 features](https://github.com/larswoudstra/Coronette/blob/main/docs/images/best_14_features.png)

Met deze 14 features gaat het dus vooralsnog eigenlijk slechts om een viertal features (hh_cmnty_cli, nohh_cmnty_cli, cli en ili) dat voor ieder van de drie gemeten dagen een relatief hoge correlatie vertoont, en de features 'tested_positive' van dag 1 en 2. Deze laatste zijn verantwoordelijk voor de hoogste pieken in de grafiek en lijken daarmee de sterkste voorspellers te zijn voor de positieve testuitslag op dag 3.

Gezien de 14 pieken wordt SelectKBest nogmaals uitgevoerd, maar nu met k=14 waarna er een nieuwe training- en testdataset gevormd wordt met de 14 features met de hoogste correlatiescores. Gekeken naar de modelprestaties kan dit aantal nog aangepast worden. Mocht het model met deze 14 features bijvoorbeeld juist underfitten, kan dit probleem opgelost worden door meer features toe te voegen.

# Model 2.5: Complexiteit verhogen

## Data analysis
--

## Data pipeline

## Model training
Er zijn verschillende combinaties en groottes van hidden layers getraind. Bij een model van 93 features blijkt een architectuur van 93x93x60x1 tot de laagste validatiekoste te komen (Validatie RMSE = 0.98). Bij een model van 14 features ligt de optimale opzet van het netwerk bij 14x5x1 (Validatie RMSE = 0.94). Vooralsnog wordt er verdergegaan met de architectuur met 14 features, aangezien deze in de laagste kosten resulteert, hoewel hiermee natuurlijk wel veel informatie verloren gaat. In de toekomst kan deze opzet aangepast worden, als blijkt dat daarmee de kosten nog verder teruggedrongen worden.

## Model evaluation
In de onderstaande tabel staan de verschillende uitgeprobeerde configuraties en de bijbehorende RMSE's op volgorde van laag naar hoog. Er is geëxperimenteerd met allerlei combinaties van netwerkarchitecturen in een poging de validatiekosten te minimaliseren. In deze tabel is duidelijk te zien dat een netwerkopzet van 14 features in de input layer, 1 hidden layer met 5 units, en 1 output layer met 1 unit tot de laagste validatiekosten leidt. Dit is dan ook het model waarmee vanaf nu verdergegaan wordt.

![Tabel met full-batch configuraties](https://github.com/larswoudstra/Coronette/blob/main/docs/images/tabel_configuraties_full_batch.png)

# Model 2.6: Batch size

## Data analysis
--

## Data pipeline
--

## Model training
Met batch size is geprobeerd het model sneller te laten leren. Met deze batch size worden de gewichten van het Neural Network per batch aangepast in plaats van per epoch, wat eerder het geval was. Door het toepassen van deze batch size en het aantal epochs verder op te schroeven, kan er mogelijk een nog lagere validation RMSE bereikt worden.

## Model evaluation
Er zijn verschillende configuraties van de batch size en het aantal epochs geprobeerd. In de tabel hieronder zijn de verschillende configuraties met de bijbehorende Validation RMSE-waarden van laag naar hoog te zien.

![Tabel met batch size configuraties](https://github.com/larswoudstra/Coronette/blob/main/docs/images/tabel_configuraties_batch_sizes.png)

Met een batch size van 70 en een aantal epochs van 700 is er een validation RMSE van 0.939 bereikt. Dit lijkt voor alsnog de beste configuratie van de batch size en het aantal epochs. Deze learning curve is hieronder te zien:

![Learning curve (14x5x1) batch_size = 70, epochs = 700](https://github.com/larswoudstra/Coronette/blob/main/docs/images/14x5x1_70batch_700epoch.png)

Omdat de loss in deze curve nog verder lijkt te dalen, is ook een configuratie met een batch size van 70 en een aantal epochs van 1000 geprobeerd. Hierbij kwam de validation loss niet lager uit: RMSE = 0.977.

# Conclusie Milestone 2
Tot nu toe geeft het fully-connected Neural Network met een 14x5x1-configuratie, een ReLU-activatiefunctie voor de hidden layer, een lineaire activatiefunctie voor de outputlayer, de 'Nadam' optimizer, een batchsize van 70 en 700 epochs de beste resultaten gebaseerd op de RMSE-metric (Validatie RMSE = 0.94). Het is dus gelukt om de gemiddelde afwijking van het daadwerkelijke percentage positieve Covid-19 tests onder 1% te krijgen.

In de toekomst zal er geprobeerd worden de kosten verder te minimaliseren, bijvoorbeeld door middel van aanpassingen van de batch size, het aantal epochs of verdere verbeteringen van de architectuur van het netwerk. Ook zal de trainingdata gesplitst worden op testdata, zodat de prestaties van het model getoetst kan worden op nieuwe data.

Daarnaast wordt er nog gekeken naar het aanpassen van de learning rate voor optimizer Nadam om te voorkomen dat het model eventueel vast blijft zitten op een zadelpunt. De validation RMSE blijft namelijk na veel verschillende configuraties hangen op een waarde tussen 0.94 en 1.  

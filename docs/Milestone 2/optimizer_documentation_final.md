Een analyse van iedere mogelijke optimizer met 100 epochs en ieders default learning rate - 0.01 voor SGD;  0.001 voor de overige optimizers - wijst een aantal interessante dingen uit. Zo blijken SGD, Adam, Adadelta en Adagrad niet de beste keuzes te zijn, omdat deze modellen al vrij snel in het trainingproces (bijvoorbeeld na 20 epochs) al niet verder lijken te leren met nog relatief hoge kosten.

Adamax, Nadam, RMSProp en Ftrl blijken na de analyse met 100 epochs daarentegen nog wel optimizers die mogelijk het overwegen waard zijn, omdat deze de laagste validatiekosten opleveren en/of nog duidelijk aan het leren zijn ten tijde van de laatste epoch.

Na een analyse met 300 epochs en nog altijd ieders default learning rate lijkt Nadam de beste keuze te zijn voor het regressieprobleem. Bij RMSProp stegen de validatiekosten namelijk naarmate het aantal epochs toenam. Ftrl leek niet veel meer te leren met 200 extra epochs, maar de kostenplot was wel erg glad. Hoewel Adamax het ook zeker niet slecht doet, lijkt dit model al gestopt te zijn met leren waar Nadams validatiekosten nog dalen. Bovendien resulteerde het model met de Nadam optimizer in de laagste kosten, en deze lijken zelfs nog een beetje te dalen na 300 epochs.

Al met al lijkt de Nadam optimizer de beste keuzen, maar Adamax en Ftrl moeten in het achterhoofd gehouden worden, aangezien deze mogelijk betere resultaten geven met andere learning rates, extra hidden layers of extra hidden nodes.


**Learning rate nog aanpassen? En moeten we dat dan ook doen bij de andere optimizers?**

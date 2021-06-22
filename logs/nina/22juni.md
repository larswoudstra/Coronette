* Meeting Tim feedback opgeschreven
* Taken opnieuw verdeeld
* Code herschreven zodat batch size, no. of epochs, k etc. makkelijker meegegeven kunnen worden
* Tabel gemaakt met verschillende geprobeerde configuraties en hun val RMSE
* Nagedacht over hoe we de kFold-functie van de Test_NN-functie kunnen scheiden, aangezien de een voor het trainen en de ander voor het testen aangeroepen moet worden. Misschien door ze in 1 functie te combineren en kFold=True/False mee te geven.. Of door een nieuwe functie te maken genaamd is_training() die, wanneer True, kFold aanroept en, wanneer False, Test_NN aanroept.
* Gediscussieerd over de inhoud van de if-name-is-main statement, uiteindelijk gemaild naar Tim en Wouter

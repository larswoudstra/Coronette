Documentatie complexiteit

Om de complexiteit te verhogen zijn we eerst aan de slag gegaan met het toevoegen van een extra hidden layer met eveneens een reLU activatiefunctie. We willen drie configuraties uitproberen: een extra hidden layer met hetzelfde aantal nodes als in de vorige layer (93), een extra hidden layer met een lager aantal nodes ten opzichte van de vorige layer (60) en een hidden layer met een hoger aantal nodes ten opzichte van de vorige layer (120).

Het baseline model (i.e., het model met de Nadam optimizer met 1 hidden layer waarin 93 nodes) gaf de volgende RMSE-waarden:
- Training RMSE: 1.100387692451477
- Validation RMSE: 1.1679387092590332

1 extra hidden layer met 93 nodes: het model lijkt lichtelijk te overfitten, zie < afbeelding complexity_1_hidden_layer_93_nodes >. BESCHRIJF VERSCHIL IN TRAINING COST TUSSEN DIT MODEL EN NADAM OPTIMIZER MET BASELINE NETWERK.
- Training RMSE: 1.1545112133026123
- Validation RMSE:  1.2482798099517822

1 extra hidden layer met 120 nodes: het model lijkt lichtelijk te overfitten, zie < afbeelding complexity_1_hidden_layer_120_nodes >. BESCHRIJF VERSCHIL IN TRAINING COST TUSSEN DIT MODEL EN NADAM OPTIMIZER MET BASELINE NETWERK.
- Training RMSE: 0.9309911131858826
- Validation RMSE: 1.0058969259262085

1 extra hidden layer met 60 nodes: het model lijkt lichtelijk te overfitten, zie < afbeelding complexity_1_hidden_layer_60_nodes >. BESCHRIJF VERSCHIL IN TRAINING COST TUSSEN DIT MODEL EN NADAM OPTIMIZER MET BASELINE NETWERK.
- Training RMSE: 0.9012
- Validation RMSE: 0.9806

1 extra hidden layer met 40 nodes: het model lijkt lichtelijk te overfitten, zie < afbeelding complexity_1_hidden_layer_40_nodes >. BESCHRIJF VERSCHIL IN TRAINING COST TUSSEN DIT MODEL EN NADAM OPTIMIZER MET BASELINE NETWERK.
- Training RMSE: 0.89
- Validation RMSE: 0.999

Nu willen we kijken of het toevoegen van nóg een extra hidden layer effect heeft op de RMSE. Omdat we bij een extra hidden layer met 60 nodes (lager dan in de vorige hidden layer) al een verlaging zagen in de RMSE, hebben we besloten om een configuratie te handhaven waarbij we in het aantal nodes willen afbouwen. Dus: 93 input nodes, een hidden layer met 93 nodes, daarna een hidden layer met 60 nodes, de laatste hidden layer met 30 nodes en een output layer met 1 node. Dit gaf de volgende resultaten:
- Training RMSE: 0.8764
- Validation RMSE: 0.87

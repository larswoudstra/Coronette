# Data analysis
Aan de hand van de loss functie van het eerste model leek het in de eerste instantie er op dat het model aan het overfitten was. Dit werd geconcludeerd uit de spikes die de gemaakte loss functie had:

< afbeelding 'losses_plus_mental_health'>

In veel van de gevallen heeft de validation data een hoge loss. Dit betekent dat het model de percentages van de validation data niet goed kan voorspellen. Om dit probleem te verhelpen is er geprobeerd om met minder features te gaan werken. De mental health features leken hiervoor het best geschikt om weg te laten, omdat deze het minste de daadwerkelijke percentages zouden kunnen voorspellen. De features die zijn weggelaten zijn: anxious, depressed, felt_isolated, worried_become_ill en worried_finances. Dit had het volgende resultaat:

< afbeelding 'losses_min_mental_health'>

In de plot is te zien dat de amplitude van de spikes is afgenomen, maar de spikes zijn nog steeds aanwezig. Omdat deze features wel enigszins invloed zouden kunnen hebben op het percentage van positieve Covid tests, lijkt het beter om de features niet helemaal uit de data te verwijderen, maar om deze features minder zwaar te laten wegen.

# Data Pipeline

# Model training
De validation data heeft in veel van de gevallen een hoge loss. Daarom wordt er gezocht naar een betere optimizer voor het Neural Network. De eerste optimizer die is gebruikt is de 'adam' optimizer. Om te bepalen welke optimizer het beste past bij de data, zijn alle optimizers van Keras geprobeerd met verschillende learning rates. 

# Model evaluation

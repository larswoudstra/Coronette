mogelijke optimizers:
<!-- - SGD (0.01): Stochastic Gradient Descent, slow and noisy
Werd nan bij laatste fold
Werd niet geplot
val_loss 61.03, VRMSE 7.81 -->


<!-- - RMSprop (0.001): Root Mean Squared, rekent met de wortel, probeert noise
                   op andere manier te verminderen, parameter-specific learning
                   rates
*100 epochs*
Validation schommelt om train heen (noisy)
Was nog bezig met daling
val_loss: 1.27, VRMSE: 1.13

*300 epochs*
Validation schommelt steeds minder om data heen
Daling stabiliseert rond 150/200 epochs
val_loss: 1.87, VRMSE: 1.37 -->


<!-- - Adam (0.001): gebruikt momentum (adaptive learning rate), efficient, RMSprop
                met momentum
Heel stijl aan het begin, daarna vrijwel horizontaal
minder noise
lijkt niet verder te leren na 20 epochs
val_loss: 1.62, VRMSE: 1.27 -->

<!-- - Adadelta (0.001): houdt rekening met decay learning rates, stochastic Gradient
                    descent
Hele smoothe afname. Sterke afname tot 40 epochs, daarna langzamer.
val_loss: 10.60, VRMSE: 3.26 -->

<!-- - Adagrad (0.001): parameter-specific learning rates
Hele smoothe afname. Sterke afname tot 20 epochs, daarna langzamer.
val_loss: 4.85, VRMSE: 2.20 -->


- Adamax (0.001): Adam met infinity norm
*100 epochs*
Heel stijl tot epoch 20, daarna langzamer.
Niet super smooth, klein beetje pieken
val_loss: 1.16, VRMSE: 1.08

*300 epochs*
Heel stijl aan t begin, daarna vrijwel horizontaal
klein beetje pieken
val_loss: 0.96, VRMSE: 0.98


**- Nadam (0.001): Adam met Nesterov momentum**
*100 epochs*
Zowel train als val schommelen heel erg
Geen enorm snelle daling aan begin
val_loss: 1.15, VRMSE: 1.07

*300 epochs*
Begin heel noisy, daarna vlakker
Validatielijn gaat boven de train lijn zitten
neemt wel nog steeds af
val_loss: 0.89, VRMSE: 0.94


- Ftrl (0.001): shallow models met grote, schaarse feature spaces
*100 epochs*
Hele smoothe lijn die echt nog aan het dalen is
val_loss: 1.54, VRMSE: 1.24

*300 epochs*
smoothe lijn, lijkt vast te lopen vanaf ongeveer 250 epochs.
nette consistente afname, maar komt niet onder de 1
val_loss: 1.51, VRMSE: 1.23

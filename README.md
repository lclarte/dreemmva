### INSTRUCTIONS

Pour faire fonctionner le code, creer le dossier **data** et telecharger les fichiers à l'adresse suivante  : 
https://challengedata.ens.fr/participants/challenges/27/
Ensuite, renommer les fichiers X_train, X_test, y_train

Les donnees sont enregistrees a une frequence de 250 hZ (500 samples a 2 secondes). Pretraitement fait dans l'article : 
* Downsampler a 128 Hz (125 Hz dans notre cas ?) 
* Filtre passe bande à 0.5 - 25 Hz


Liens pertinents : 
- Article original : https://www.nature.com/articles/s41598-018-21495-7.pdf
- Neural Networks for EEG : https://arxiv.org/abs/1703.05051 & https://github.com/TNTLFreiburg/braindecode
- Review of class. algorithms for EEG-based BCI : https://hal.inria.fr/inria-00134950/document
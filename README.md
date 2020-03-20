### TODO

première idée à la volée : on slip chaque sample de 2 sec, on fait un transformée de Fourier localisée en fréquence pour les différentes composantes (beta, theta, sleep spindle, Kcomplex etc.). Ensuite une ben on classifie linéairement de bases sur les coefficients (Faire une random forest ?)

1) Implementation du modele linéaire : 
    - Pouvoir calculer les spindles : (https://raphaelvallat.com/spindles.html) 
        * Regarder les differences dans le nombre de spindles entre homme et femme
    - Pouvoir calculer la power spectrum density en utilisant la methode de Welch
        * Regarder differences Hommes et Femmes


### INSTRUCTIONS

Pour faire fonctionner le code, creer le dossier **data** et telecharger les fichiers à l'adresse suivante: 
https://challengedata.ens.fr/participants/challenges/27/
Ensuite, renommer les fichiers X_train, X_test, y_train

Les donnees sont enregistrees a une frequence de 250 hZ (500 samples a 2 secondes). Pretraitement fait dans l'article : 
* Downsampler a 128 Hz (125 Hz dans notre cas ?) 
* Filtre passe bande à 0.5 - 25 Hz

Liens pertinents : 
- Article original : https://www.nature.com/articles/s41598-018-21495-7.pdf
    * Rque : methode utilisee pour calculer le power spectrum : Welch's method sur des epochs "half-overlapping" de 10 seconds 
- EEG-Based Age and Gender Prediction Using Deep BLSTM-LSTM Network Model
    * Both above articles stipulate that beta waves are better for gender prediction
- Braindecode : neural Networks for EEG : https://arxiv.org/abs/1703.05051 & https://github.com/TNTLFreiburg/braindecode
- Review of class. algorithms for EEG-based BCI : https://hal.inria.fr/inria-00134950/document
- Muller et al., ML techniques for BCI (2004) : http://doc.ml.tu-berlin.de/bbci/publications/MueKraDorCurBla04.pdf
- Blankertz et al., Classifying single trial EEG (2002): https://papers.nips.cc/paper/2030-classifying-single-trial-eeg-towards-brain-computer-interfacing.pdf
- Del Rio-Portilla et al., Sex differences in EEG in adult gonadectomized rats (...) (1997)
- Latta et al., Sex differences in Delta and Alpha EEG activities in healthy older adults (2018)
- Puranik et al., Elementary Time Frequency Analysis of EEG sig. proc. (2018)
- Logistic Regression in Rare Events Data, King, Zen, 2001
    * Justifies the class weights in the imbalanced dataset (sklearn function class_weight.compute_class_weight)
- DOSED : A deep learning approach to detect multiple sleep micro-events : https://scihub.bban.top/https://doi.org/10.1016/j.jneumeth.2019.03.017
- A deep learning architecture for temporal sleep 
- https://github.com/tevisgehr/EEG-Classification
    * Preprocessing des EEG en faisant des FFT sur des fenetres glissantes de 1 secondes puis frequency binning
- https://github.com/keras-team/keras-tuner
    * Libraire pour tuner les hyperparametres dans un modele sur Keras
- https://github.com/Dreem-Organization/dosed
    * DOSED : Detection of spindles and K-complexes
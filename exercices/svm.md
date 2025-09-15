# 🏋️ Exercice SVM : Choix de noyau selon les données (20min)

## Contexte
Vous disposez de 4 jeux de données différents avec des structures géométriques variées.

## Consigne
Pour chaque jeu de données ci-dessous, **justifiez le choix du noyau** le plus approprié et **estimez les hyperparamètres** :

### Dataset A : Classification de textes
- **Données** : 10,000 documents représentés par des vecteurs TF-IDF sparse (50,000 dimensions)
- **Classes** : spam vs non-spam
- **Caractéristiques** : haute dimension, données creuses

**Questions :**
1. Quel noyau choisir et pourquoi ?
2. Quelle valeur de C recommandez-vous ?

### Dataset B : Reconnaissance de formes géométriques  
- **Données** : coordonnées (x,y) de points dans le plan
- **Classes** : intérieur vs extérieur d'une ellipse
- **Taille** : 1,000 points

**Questions :**
1. Quel noyau utiliser ?
2. Comment ajuster le paramètre γ ?

### Dataset C : Données financières
- **Données** : 20 ratios financiers d'entreprises
- **Classes** : défaut vs non-défaut de paiement  
- **Contrainte** : interprétabilité requise

**Questions :**
1. Noyau recommandé ?
2. Justifiez votre choix par rapport à la contrainte métier.

### Dataset D : Données mixtes complexes
- **Données** : relation y = sin(x₁) × x₂³ + bruit
- **Classes** : y > 0 vs y ≤ 0
- **Défi** : interaction non-linéaire complexe

**Questions :**
1. Stratégie de choix de noyau ?
2. Méthode de validation recommandée ?

## Solutions attendues

### Dataset A → **Noyau linéaire**
- Haute dimension → souvent linéairement séparable
- Rapidité de calcul essentielle
- C modéré (1.0) pour éviter le sur-ajustement

### Dataset B → **Noyau RBF**  
- Structure non-linéaire évidente
- γ ≈ 1/variance_des_données
- Validation croisée pour ajustement fin

### Dataset C → **Noyau linéaire**
- Interprétabilité prioritaire
- Les coefficients restent analysables
- C faible pour régularisation

### Dataset D → **Noyau polynomial ou RBF**
- Test des deux avec validation croisée
- Polynomial degré 3-4 ou RBF avec γ ajusté
- Attention au sur-ajustement

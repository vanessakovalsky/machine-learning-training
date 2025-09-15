# 🏗️ Exercice réseaux de neurones : Architecture pour problème donné (15min)

## Contexte
Vous devez concevoir des architectures de réseaux de neurones pour différents types de problèmes.

## Problèmes à résoudre

### Problème 1 : Classification d'images (MNIST)
- **Données** : images 28×28 pixels en niveaux de gris
- **Classes** : chiffres 0-9 (10 classes)
- **Contrainte** : architecture simple pour débutant

**Concevez :**
1. Architecture du réseau (nombre de couches, neurones)
2. Fonctions d'activation pour chaque couche
3. Fonction de coût appropriée

### Problème 2 : Régression - Prédiction de prix immobilier
- **Données** : 15 caractéristiques numériques normalisées
- **Sortie** : prix (valeur continue)
- **Objectif** : minimiser l'erreur quadratique

**Concevez :**
1. Architecture adaptée à la régression
2. Choix de la fonction de sortie
3. Métrique d'évaluation

### Problème 3 : Classification binaire déséquilibrée
- **Données** : détection de fraude (99% non-fraude, 1% fraude)
- **Variables** : 50 caractéristiques de transaction
- **Défi** : classes très déséquilibrées

**Concevez :**
1. Architecture robuste au déséquilibrement
2. Stratégie de gestion des classes déséquilibrées
3. Métriques d'évaluation appropriées

## Solutions recommandées

### Solution 1 - MNIST
```
Architecture :
- Entrée : 784 neurones (28×28 aplati)
- Couche cachée 1 : 128 neurones + ReLU
- Couche cachée 2 : 64 neurones + ReLU  
- Sortie : 10 neurones + Softmax

Fonction de coût : Entropie croisée catégorielle
Optimiseur : Adam, lr=0.001
```

### Solution 2 - Régression immobilière  
```
Architecture :
- Entrée : 15 neurones
- Couche cachée 1 : 64 neurones + ReLU
- Couche cachée 2 : 32 neurones + ReLU
- Sortie : 1 neurone + linéaire

Fonction de coût : MSE (Mean Squared Error)
Métrique : RMSE, MAE
```

### Solution 3 - Détection de fraude
```
Architecture :
- Entrée : 50 neurones  
- Couche cachée 1 : 100 neurones + ReLU + Dropout(0.3)
- Couche cachée 2 : 50 neurones + ReLU + Dropout(0.2)
- Sortie : 1 neurone + Sigmoid

Stratégies spéciales :
- Pondération des classes (class_weight)
- Seuil de décision ajusté
- Métriques : Précision, Rappel, F1-Score, AUC-ROC
```

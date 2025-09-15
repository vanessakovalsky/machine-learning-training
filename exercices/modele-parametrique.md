## EXERCICE : Application sur dataset simple (25min)

### Dataset : Classification d'espèces d'iris (version simplifiée)

Nous utilisons seulement 2 variables et 2 classes pour faciliter la visualisation.

**Variables :**
- X1 : Longueur des sépales (cm)
- X2 : Largeur des sépales (cm)

**Classes :**
- Setosa (S)
- Versicolor (V)

**Données d'apprentissage :**

| Obs | X1 | X2 | Classe |
|-----|----|----|--------|
| 1   | 5.1| 3.5| S      |
| 2   | 4.9| 3.0| S      |
| 3   | 4.7| 3.2| S      |
| 4   | 4.6| 3.1| S      |
| 5   | 5.0| 3.6| S      |
| 6   | 7.0| 3.2| V      |
| 7   | 6.4| 3.2| V      |
| 8   | 6.9| 3.1| V      |
| 9   | 5.5| 2.3| V      |
| 10  | 6.5| 2.8| V      |

### Étapes de l'exercice

#### Étape 1 : Calcul des statistiques descriptives (5min)

1. **Calculez les moyennes par classe :**
   - μ̂_S = ?
   - μ̂_V = ?

2. **Calculez les probabilités a priori :**
   - π̂_S = ?
   - π̂_V = ?

#### Étape 2 : Estimation de la matrice de covariance (8min)

1. **Calculez les matrices de covariance intra-classe :**
   - S_S (matrice 2×2 pour Setosa)
   - S_V (matrice 2×2 pour Versicolor)

2. **Calculez la matrice de covariance poolée :**
   - Σ̂ = ?

#### Étape 3 : Construction des fonctions discriminantes (7min)

1. **Calculez les coefficients des fonctions discriminantes**
2. **Déterminez l'équation de la frontière de décision**

#### Étape 4 : Prédiction et évaluation (5min)

**Nouvelles observations à classifier :**
- Point A : (5.5, 3.0)
- Point B : (6.0, 2.5)

1. Calculez δ_S(x) et δ_V(x) pour chaque point
2. Déterminez la classe prédite
3. Calculez les probabilités d'appartenance

### Solution détaillée

#### Étape 1 - Solutions :
```
μ̂_S = (4.86, 3.28)
μ̂_V = (6.26, 2.92)
π̂_S = π̂_V = 0.5
```

#### Étape 2 - Matrice de covariance poolée :
```
Σ̂ = [0.685  0.042]
     [0.042  0.188]
```

#### Étape 3 - Fonctions discriminantes :
```
δ_S(x) = 1.89×X1 + 0.94×X2 - 5.83
δ_V(x) = 2.13×X1 + 1.12×X2 - 7.21
```

#### Étape 4 - Prédictions :
```
Point A (5.5, 3.0):
- δ_S(A) = 5.58, δ_V(A) = 5.83
- Classe prédite: Versicolor
- P(V|A) = 0.56, P(S|A) = 0.44

Point B (6.0, 2.5):
- δ_S(B) = 5.48, δ_V(B) = 6.88
- Classe prédite: Versicolor  
- P(V|B) = 0.80, P(S|B) = 0.20
```

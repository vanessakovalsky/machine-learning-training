# EXERCICE PRATIQUE : Calcul de probabilités (25min)

### Dataset : Classification de spams

Un fournisseur d'email veut classifier automatiquement les emails comme SPAM ou HAM (légitime).

**Variables :**
- Contient "gratuit" : Oui/Non
- Contient "urgent" : Oui/Non  
- Nombre d'exclamations : 0, 1-2, 3+

**Données d'apprentissage :**

| Email | Gratuit | Urgent | Exclamations | Classe |
|-------|---------|--------|--------------|---------|
| 1     | Oui     | Non    | 3+          | SPAM    |
| 2     | Non     | Non    | 0           | HAM     |
| 3     | Oui     | Oui    | 1-2         | SPAM    |
| 4     | Non     | Non    | 0           | HAM     |
| 5     | Non     | Non    | 1-2         | HAM     |
| 6     | Oui     | Non    | 3+          | SPAM    |
| 7     | Non     | Oui    | 0           | HAM     |
| 8     | Oui     | Oui    | 3+          | SPAM    |

### Étapes de l'exercice

#### Étape 1 : Calcul des probabilités a priori (3min)
1. P(SPAM) = ?
2. P(HAM) = ?

#### Étape 2 : Calcul des probabilités conditionnelles (10min)

**Pour la classe SPAM :**
1. P(Gratuit=Oui|SPAM) = ?
2. P(Urgent=Oui|SPAM) = ?  
3. P(Exclamations=3+|SPAM) = ?

**Pour la classe HAM :**
1. P(Gratuit=Oui|HAM) = ?
2. P(Urgent=Oui|HAM) = ?
3. P(Exclamations=3+|HAM) = ?

#### Étape 3 : Classification d'un nouvel email (8min)

**Nouvel email :** Contient "gratuit"=Oui, "urgent"=Non, Exclamations=1-2

1. Calculez P(X|SPAM) × P(SPAM)
2. Calculez P(X|HAM) × P(HAM)
3. Déterminez la classe prédite
4. Calculez les probabilités a posteriori

#### Étape 4 : Effet du lissage de Laplace (4min)

Recalculez avec α = 1 les probabilités qui étaient nulles et observez l'impact.

### Solutions détaillées

#### Étape 1 - Probabilités a priori :
```
P(SPAM) = 4/8 = 0.5
P(HAM) = 4/8 = 0.5
```

#### Étape 2 - Probabilités conditionnelles :

**Classe SPAM (4 emails) :**
```
P(Gratuit=Oui|SPAM) = 3/4 = 0.75
P(Urgent=Oui|SPAM) = 2/4 = 0.5
P(Exclamations=3+|SPAM) = 3/4 = 0.75
P(Exclamations=1-2|SPAM) = 1/4 = 0.25
```

**Classe HAM (4 emails) :**
```
P(Gratuit=Oui|HAM) = 0/4 = 0
P(Urgent=Oui|HAM) = 1/4 = 0.25
P(Exclamations=3+|HAM) = 0/4 = 0
P(Exclamations=1-2|HAM) = 1/4 = 0.25
```

#### Étape 3 - Classification :

**Email : (Gratuit=Oui, Urgent=Non, Exclamations=1-2)**

```
P(X|SPAM) × P(SPAM) = 0.75 × 0.5 × 0.25 × 0.5 = 0.046875
P(X|HAM) × P(HAM) = 0 × 0.75 × 0.25 × 0.5 = 0
```

**Problème :** P(Gratuit=Oui|HAM) = 0 → résultat = 0

#### Étape 4 - Avec lissage de Laplace (α=1) :

```
P(Gratuit=Oui|HAM) = (0+1)/(4+2) = 1/6 ≈ 0.167
P(Urgent=Non|HAM) = (3+1)/(4+2) = 4/6 ≈ 0.667

P(X|HAM) × P(HAM) = 0.167 × 0.667 × 0.25 × 0.5 ≈ 0.014

Probabilités finales :
P(SPAM|X) = 0.047 / (0.047 + 0.014) ≈ 0.77
P(HAM|X) = 0.014 / (0.047 + 0.014) ≈ 0.23

Classification : SPAM
```

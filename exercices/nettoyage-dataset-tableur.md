# 🎯 Nettoyer un dataset et préparer une analyse exploratoire

## 📋 Prérequis
- Excel (2016+) ou Google Sheets
- Calculatrice (pour vérifications)
- Connaissance des fonctions de base Excel (MOYENNE, MEDIANE, NB.SI, etc.)


## 🏠 **Exercice 1 : Nettoyage d'un Dataset Immobilier (25 minutes)**

### 🎯 Objectif
Nettoyer un dataset de ventes immobilières contenant des données manquantes et aberrantes

### 📊 Dataset fourni
**Fichier :** `ventes_immobilieres.xlsx`

**Colonnes :**
- `ID_Vente` : Identifiant unique
- `Prix_Vente` : Prix en euros (contient des valeurs manquantes et aberrantes)
- `Surface_m2` : Surface habitable (valeurs manquantes)
- `Nb_Chambres` : Nombre de chambres (quelques valeurs manquantes)
- `Age_Bien` : Âge du bien en années
- `Type_Bien` : Appartement/Maison
- `Quartier` : Zone géographique
- `Distance_Centre` : Distance au centre-ville en km

### 🔧 Tâches à réaliser

#### **1. Charger et examiner les données (5 min)**
- Ouvrir le fichier dans Excel/Google Sheets
- Identifier le nombre total de lignes
- Créer un onglet "Analyse_Initiale"
- Noter vos observations dans une zone de commentaires

**Questions à se poser :**
- Combien d'enregistrements au total ?
- Quelles colonnes semblent problématiques au premier coup d'œil ?

#### **2. Identifier les données manquantes (5 min)**
Créer un tableau de synthèse dans l'onglet "Analyse_Initiale" :

| Colonne | Nb Valeurs Manquantes | % Manquant | Formule Excel |
|---------|----------------------|------------|---------------|
| Prix_Vente | `=NBVAL(B:B)` | `=NBVAL(B:B)/LIGNES()*100` | `=NB.VIDE(B:B)` |
| Surface_m2 | | | |
| Nb_Chambres | | | |

**Instructions :**
- Utilisez `NBVAL()` pour compter les cellules non vides
- Utilisez `NB.VIDE()` pour compter les cellules vides
- Calculez les pourcentages

#### **3. Traiter les valeurs manquantes - 2 méthodes (8 min)**

**Créer 2 nouveaux onglets :**

**Onglet "Methode1_Moyenne"**
- Remplacer les valeurs manquantes de `Prix_Vente` par la moyenne
- Formule : `=SI(ESTVIDE(B2);MOYENNE($B$2:$B$1000);B2)`
- Remplacer les valeurs manquantes de `Surface_m2` par la médiane
- Formule : `=SI(ESTVIDE(C2);MEDIANE($C$2:$C$1000);C2)`

**Onglet "Methode2_Regression"**
- Pour `Prix_Vente` manquant : utiliser Surface_m2 comme prédicteur
- Formule approximative : `Prix = Surface * Prix_moyen_par_m2`
- Calculer : `=SI(ESTVIDE(B2);C2*MOYENNE($B$2:$B$1000)/MOYENNE($C$2:$C$1000);B2)`

#### **4. Détecter les outliers - 2 techniques (4 min)**

**Technique 1 : Méthode IQR (Interquartile Range)**
```
Q1 = QUARTILE(Prix_Vente; 1)
Q3 = QUARTILE(Prix_Vente; 3)
IQR = Q3 - Q1
Limite_Basse = Q1 - 1.5*IQR
Limite_Haute = Q3 + 1.5*IQR
```

Colonne "Outlier_IQR" : `=SI(OU(B2<Limite_Basse;B2>Limite_Haute);"OUTLIER";"OK")`

**Technique 2 : Méthode Z-Score**
```
Moyenne = MOYENNE(Prix_Vente)
Ecart_Type = ECARTYPE(Prix_Vente)
Z_Score = (Valeur - Moyenne) / Ecart_Type
```

Colonne "Outlier_ZScore" : `=SI(ABS((B2-MOYENNE($B:$B))/ECARTYPE($B:$B))>3;"OUTLIER";"OK")`

#### **5. Normalisation ET Standardisation (3 min)**

**Normalisation (Min-Max) :**
`Prix_Normalise = (Prix - MIN(Prix)) / (MAX(Prix) - MIN(Prix))`

**Standardisation (Z-Score) :**
`Prix_Standardise = (Prix - MOYENNE(Prix)) / ECARTYPE(Prix)`

Formules Excel :
- Normalisation : `=(B2-MIN($B:$B))/(MAX($B:$B)-MIN($B:$B))`
- Standardisation : `=(B2-MOYENNE($B:$B))/ECARTYPE($B:$B)`


## 📊 **Exercice 2 : Analyse Exploratoire - Dataset Marketing (30 minutes)**

### 🎯 Objectif
Analyser un dataset de campagne marketing pour identifier les variables prédictives

### 📊 Dataset fourni
**Fichier :** `campagne_marketing.xlsx`

**Colonnes :**
- `Age` : Âge du client
- `Salaire` : Salaire annuel
- `Nb_Enfants` : Nombre d'enfants
- `Anciennete_Client` : Années en tant que client
- `Nb_Achats_Precedents` : Historique d'achats
- `Canal_Acquisition` : Email/SMS/Pub/Direct
- `Categorie_Socio` : A/B/C
- `A_Achete` : Variable cible (0/1)

### 🔧 Tâches à réaliser

#### **1. Analyse univariée complète (8 min)**

**Créer un onglet "Analyse_Univariee"**

Pour chaque variable numérique, calculer :

| Variable | Moyenne | Médiane | Écart-type | Min | Max | Q1 | Q3 |
|----------|---------|---------|------------|-----|-----|----|----|
| Age | `=MOYENNE(B:B)` | `=MEDIANE(B:B)` | `=ECARTYPE(B:B)` | | | | |
| Salaire | | | | | | | |

**Pour variables catégorielles, créer des tableaux de fréquence :**
- Utiliser `NB.SI()` pour compter les occurrences
- Calculer les pourcentages

#### **2. Matrice de corrélation (8 min)**

**Créer un onglet "Correlations"**

Matrice de corrélation entre variables numériques :

| | Age | Salaire | Nb_Enfants | Anciennete | Nb_Achats | A_Achete |
|---|-----|---------|------------|------------|-----------|----------|
| Age | 1 | `=COEFFICIENT.CORRELATION(Age;Salaire)` | | | | |
| Salaire | | 1 | | | | |

**Formule générale :** `=COEFFICIENT.CORRELATION(plage1;plage2)`

**Interprétation :**
- Corrélation > 0.7 : forte corrélation positive
- Corrélation < -0.7 : forte corrélation négative
- |Corrélation| < 0.3 : corrélation faible

#### **3. Identifier la multicolinéarité (4 min)**

Dans l'onglet "Correlations", identifier :
- Paires de variables avec |corrélation| > 0.8
- Créer une liste des variables multicolinéaires à surveiller

#### **4. Sélection de features - 3 techniques (7 min)**

**Technique 1 : Corrélation avec la variable cible**
Calculer corrélation de chaque variable avec `A_Achete` :
- `=COEFFICIENT.CORRELATION(Age;A_Achete)`
- Classer par corrélation absolue décroissante

**Technique 2 : Analyse de variance par groupe**
Pour les acheteurs vs non-acheteurs :
- Moyenne Age acheteurs : `=MOYENNE.SI(A_Achete;1;Age)`
- Moyenne Age non-acheteurs : `=MOYENNE.SI(A_Achete;0;Age)`
- Différence : variables avec grande différence sont importantes

**Technique 3 : Information Gain approximatif**
- Calculer entropie par sous-groupe
- Variables qui "séparent" mieux les classes

#### **5. Recommandations variables finales (2 min)**

Créer un tableau de synthèse :

| Variable | Corrélation Cible | Variance Entre Groupes | Score Final | Recommandation |
|----------|-------------------|------------------------|-------------|----------------|
| Age | 0.45 | Élevée | ⭐⭐⭐ | À garder |
| Salaire | 0.62 | Élevée | ⭐⭐⭐⭐ | À garder |

#### **6. Rapport de synthèse avec visualisations (1 min)**

**Créer un onglet "Rapport_Final"**

**Résumé exécutif :**
- Qualité des données : X% de valeurs manquantes traitées
- Variables recommandées : [liste]
- Variables à exclure : [liste + justification]
- Outliers détectés : X% des observations

**Graphiques recommandés (à créer manuellement) :**
- Histogrammes des variables principales
- Graphiques en barres pour variables catégorielles
- Nuage de points pour relations importantes

---

## 📈 **Critères d'évaluation**

### Exercice 1 (50 points)
- **Qualité du nettoyage (20 pts)** : Traitement approprié des valeurs manquantes
- **Détection outliers (15 pts)** : Application correcte des 2 méthodes
- **Normalisation (10 pts)** : Formules correctes et interprétation
- **Justification choix (5 pts)** : Argumentation des décisions prises

### Exercice 2 (50 points)
- **Analyse univariée (15 pts)** : Complétude et précision
- **Matrice corrélation (15 pts)** : Calculs corrects et interprétation
- **Sélection features (15 pts)** : Application des 3 techniques
- **Rapport synthèse (5 pts)** : Clarté et recommandations pertinentes

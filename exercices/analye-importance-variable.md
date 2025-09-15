# EXERCICE : Analyse d'importance de variables (20min)

### Contexte
Une banque utilise Random Forest pour prédire le défaut de paiement de ses clients. Vous analysez un modèle entraîné avec les résultats suivants.

### Variables et importance

**Dataset :** 10,000 clients, 15 variables

| Variable | Type | Importance (Gini) | Importance (Permutation) |
|----------|------|-------------------|--------------------------|
| Score_credit | Continue | 0.142 | 0.089 |
| Revenu_annuel | Continue | 0.098 | 0.076 |
| Ratio_endettement | Continue | 0.089 | 0.082 |
| Anciennete_emploi | Continue | 0.076 | 0.054 |
| Age | Continue | 0.065 | 0.041 |
| Montant_demande | Continue | 0.058 | 0.048 |
| Nb_enfants | Discrète | 0.045 | 0.031 |
| Statut_marital | Catégorielle | 0.042 | 0.035 |
| Type_logement | Catégorielle | 0.038 | 0.028 |
| Secteur_activite | Catégorielle | 0.035 | 0.025 |
| Region | Catégorielle | 0.032 | 0.019 |
| Formation | Catégorielle | 0.028 | 0.021 |
| Sexe | Binaire | 0.025 | 0.018 |
| Nb_comptes_banque | Discrète | 0.022 | 0.016 |
| Historique_retards | Binaire | 0.305 | 0.217 |

### Performances du modèle

```
Erreur Out-of-Bag : 0.086 (8.6%)
Nombre d'arbres : 500
Variables par nœud (m) : 4 (√15 ≈ 4)
```

### Questions d'analyse

#### Question 1 : Interprétation des importances (5min)
1. Quelles sont les 3 variables les plus importantes selon chaque méthode ?
2. Pourquoi y a-t-il des différences entre les deux méthodes ?
3. Que suggère l'importance très élevée d'`Historique_retards` ?

#### Question 2 : Recommandations métier (7min)
1. Sur quelles variables la banque devrait-elle se concentrer pour améliorer son modèle de scoring ?
2. Y a-t-il des variables surprenantes par leur faible importance ?
3. Proposez 3 nouvelles variables qui pourraient être pertinentes.

#### Question 3 : Optimisation du modèle (5min)
1. Le nombre de variables par nœud (m=4) vous semble-t-il optimal ?
2. Comment pourriez-vous valider ce choix ?
3. Quelle stratégie pour traiter les variables peu importantes ?

#### Question 4 : Validation et déploiement (3min)
1. L'erreur OOB de 8.6% est-elle suffisante pour évaluer le modèle ?
2. Quelles étapes supplémentaires recommandez-vous avant mise en production ?

### Solutions détaillées

#### Solution 1 : Interprétation
```
Top 3 (Gini) : Historique_retards (30.5%), Score_credit (14.2%), Revenu_annuel (9.8%)
Top 3 (Permutation) : Historique_retards (21.7%), Score_credit (8.9%), Ratio_endettement (8.2%)

Différences : 
- Méthode Gini favorise les variables continues et à nombreuses modalités
- Permutation donne une importance plus "réelle" de l'impact prédictif
- Historique_retards est dominante → forte capacité prédictive

Interprétation : Les antécédents de paiement sont le facteur prédictif principal du défaut
```

#### Solution 2 : Recommandations
```
Concentration prioritaire :
1. Historique_retards : Variable la plus prédictive
2. Score_credit : Information synthétique cruciale  
3. Ratio_endettement : Indicateur de capacité de remboursement

Variables surprenantes (faible importance) :
- Sexe : Cohérent (discrimination interdite)
- Region : Peut indiquer faible variation géographique
- Formation : Impact indirect via revenu/emploi

Nouvelles variables potentielles :
1. Évolution récente du revenu (6 derniers mois)
2. Nombre de crédits actifs simultanés
3. Montant moyen des transactions récentes
```

#### Solution 3 : Optimisation
```
m = 4 : Semble correct pour p=15 (√15 ≈ 4)

Validation du choix :
- Tester m ∈ {2,3,4,5,6} par validation croisée
- Optimiser par recherche sur grille
- Observer l'erreur OOB en fonction de m

Stratégie variables peu importantes :
1. Sélection de variables : garder top 8-10 variables
2. Test d'impact : modèle réduit vs complet
3. Analyse coût/bénéfice de la collecte de données
```

#### Solution 4 : Validation
```
Erreur OOB 8.6% : Bon indicateur mais insuffisant seul

Étapes supplémentaires :
1. Validation sur échantillon test indépendant
2. Analyse de stabilité temporelle (backtesting)
3. Tests de robustesse sur sous-populations
4. Calibration des probabilités prédites
5. Tests d'équité/non-discrimination
6. Documentation et gouvernance du modèle
```

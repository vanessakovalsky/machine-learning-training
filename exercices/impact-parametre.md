# EXERCICE : Impact des paramètres (25min)

### Contexte
Vous optimisez un modèle Gradient Boosting pour prédire les ventes immobilières. Voici les résultats d'une recherche de paramètres.

### Résultats d'expérimentation

**Dataset :** 5,000 transactions immobilières, 20 variables explicatives

| Paramètres | RMSE Train | RMSE Valid | Temps (s) | Surapprentissage |
|------------|------------|------------|-----------|------------------|
| lr=0.1, depth=3, n=100 | 45,2 | 52,8 | 12 | Faible |
| lr=0.1, depth=3, n=500 | 38,1 | 49,2 | 58 | Modéré |
| lr=0.1, depth=3, n=1000 | 32,4 | 48,9 | 115 | Modéré |
| lr=0.1, depth=6, n=100 | 38,9 | 51,5 | 28 | Modéré |
| lr=0.1, depth=6, n=500 | 22,6 | 47,3 | 142 | Fort |
| lr=0.3, depth=3, n=100 | 42,1 | 53,9 | 12 | Faible |
| lr=0.3, depth=3, n=500 | 28,7 | 51,1 | 58 | Fort |
| lr=0.05, depth=3, n=1000 | 35,8 | 47,8 | 230 | Faible |
| lr=0.01, depth=3, n=2000 | 38,2 | 47,1 | 920 | Très faible |

### Évolution temporelle (lr=0.1, depth=3)

| Itération | RMSE Train | RMSE Valid |
|-----------|------------|------------|
| 50 | 49,8 | 54,2 |
| 100 | 45,2 | 52,8 |
| 200 | 41,3 | 50,9 |
| 300 | 39,1 | 49,8 |
| 400 | 37,8 | 49,4 |
| 500 | 38,1 | 49,2 |
| 750 | 34,2 | 49,1 |
| 1000 | 32,4 | 48,9 |

### Questions d'analyse

#### Question 1 : Analyse des paramètres (8min)
1. Quel est l'effet du learning rate sur les performances ?
2. Comment la profondeur des arbres influence-t-elle le surapprentissage ?
3. Quel est le meilleur compromis performance/temps de calcul ?

#### Question 2 : Détection du surapprentissage (7min)
1. À partir de quelle itération observe-t-on du surapprentissage dans l'évolution temporelle ?
2. Quelle stratégie d'early stopping recommandez-vous ?
3. Comment expliquer que le surapprentissage puisse être "modéré" même avec beaucoup d'itérations ?

#### Question 3 : Recommandations finales (6min)
1. Quels paramètres recommandez-vous pour la production ?
2. Comment améliorer encore les performances ?
3. Quels tests supplémentaires effectueriez-vous ?

#### Question 4 : Stratégie d'optimisation (4min)
1. Dans quel ordre optimiser les paramètres ?
2. Quelle approche pour des contraintes de temps strictes ?

### Solutions détaillées

#### Solution 1 : Analyse des paramètres
```
Learning rate :
- lr=0.01 : Apprentissage très lent, sous-apprentissage
- lr=0.05 : Bon équilibre, peu de surapprentissage  
- lr=0.1 : Standard, performances correctes
- lr=0.3 : Apprentissage rapide mais instable

Profondeur arbres :
- depth=3 : Équilibre biais/variance optimal
- depth=6 : Meilleure capacité mais surapprentissage plus fort

Meilleur compromis : lr=0.05, depth=3, n=1000
→ RMSE=47.8, temps raisonnable (230s), faible surapprentissage
```

#### Solution 2 : Surapprentissage
```
Surapprentissage observé : Après itération 300-400
→ RMSE train continue à baisser mais RMSE valid stagne

Early stopping recommandé :
- Patience = 50-100 itérations
- Monitoring de RMSE valid
- Arrêt si pas d'amélioration > seuil

Surapprentissage modéré expliqué par :
- Learning rate faible (0.1) régularise naturellement
- Arbres peu profonds limitent la complexité
```

#### Solution 3 : Recommandations production
```
Paramètres recommandés :
- lr=0.05, depth=3, n=1000 avec early stopping
- Alternative rapide : lr=0.1, depth=3, n=500

Améliorations possibles :
1. Feature engineering (interactions, transformations)
2. Ensemble avec Random Forest
3. Optimisation bayésienne des hyperparamètres
4. Régularisation L1/L2
5. Validation croisée stratifiée

Tests supplémentaires :
- Stabilité sur données temporelles
- Performance par sous-groupes
- Analyse des résidus
```

#### Solution 4 : Stratégie optimisation
```
Ordre d'optimisation :
1. max_depth (impact fort sur complexité)
2. learning_rate (compromis vitesse/qualité)  
3. n_estimators (avec early stopping)
4. Paramètres régularisation (subsampling, etc.)

Contraintes temps :
- Grid search limité sur paramètres clés
- Validation simple (hold-out) au lieu de CV
- Early stopping agressif
- Parallélisation si possible
```

---

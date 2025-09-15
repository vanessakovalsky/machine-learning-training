## EXERCICE : Interprétation de résultats (20min)

### Contexte
Une entreprise de marketing veut prédire la probabilité qu'un client réponde positivement à une campagne publicitaire.

**Variables :**
- Y : Réponse (1=Oui, 0=Non)
- X₁ : Âge (années)
- X₂ : Revenu (milliers d'euros)
- X₃ : Sexe (1=Homme, 0=Femme)

### Résultats de la régression logistique

```
Modèle estimé :
logit(p) = -2.104 + 0.027×Âge + 0.035×Revenu + 0.421×Sexe

Coefficients :
                Estimate  Std.Error  z-value  Pr(>|z|)
(Intercept)     -2.104      0.234   -8.99    < 0.001 ***
Âge              0.027      0.008    3.38     0.001 **
Revenu           0.035      0.012    2.92     0.004 **
Sexe             0.421      0.186    2.26     0.024 *

Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Log-Likelihood: -298.7
AIC: 605.4
Pseudo R²: 0.156
```

### Questions d'interprétation

#### Question 1 : Significativité des variables (3min)
1. Quelles variables sont significatives au seuil de 5% ?
2. Que signifie le coefficient de l'âge ?

#### Question 2 : Calcul des odds ratios (5min)
1. Calculez l'odds ratio pour chaque variable
2. Interprétez l'odds ratio du revenu

#### Question 3 : Prédiction (7min)
Calculez la probabilité de réponse positive pour :
- Client A : Femme, 35 ans, 45k€ de revenu
- Client B : Homme, 50 ans, 60k€ de revenu

#### Question 4 : Comparaison des profils (5min)
1. Quel client a la plus forte probabilité de réponse ?
2. Calculez la différence en points de pourcentage

### Solutions

#### Solution 1 :
```
Variables significatives au seuil 5% : Âge, Revenu, Sexe (toutes)
Coefficient âge (0.027) : chaque année d'âge supplémentaire 
augmente les log-odds de 0.027
```

#### Solution 2 :
```
Odds ratios :
- Âge : exp(0.027) = 1.027 (+2.7% par année)
- Revenu : exp(0.035) = 1.036 (+3.6% par 1k€)
- Sexe : exp(0.421) = 1.524 (+52.4% pour les hommes)
```

#### Solution 3 :
```
Client A (F, 35, 45k€) :
z = -2.104 + 0.027×35 + 0.035×45 + 0.421×0 = -0.1315
p = 1/(1+exp(0.1315)) = 0.467 = 46.7%

Client B (H, 50, 60k€) :
z = -2.104 + 0.027×50 + 0.035×60 + 0.421×1 = 0.813
p = 1/(1+exp(-0.813)) = 0.692 = 69.2%
```

#### Solution 4 :
```
Client B a la plus forte probabilité (69.2% vs 46.7%)
Différence : 69.2% - 46.7% = 22.5 points de pourcentage
```

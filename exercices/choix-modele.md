# EXERCICE : Choix de modèle selon contexte (15min)

### Contexte

Vous êtes data scientist dans différentes entreprises. Pour chaque situation, déterminez le type de régression logistique approprié et justifiez votre choix.

### Situations

#### Situation 1 : E-commerce
**Variable à prédire :** Catégorie de produit acheté
**Modalités :** {Électronique, Mode, Maison, Sport, Livres}
**Variables explicatives :** Âge, Sexe, Revenu, Historique d'achat

**Questions :**
1. Quel modèle choisir ?
2. Quelle serait la classe de référence ?

#### Situation 2 : Hôpital
**Variable à prédire :** Intensité de la douleur post-opératoire
**Modalités :** {Nulle, Légère, Modérée, Forte, Extrême}
**Variables explicatives :** Type d'opération, Âge, Poids, Antécédents

**Questions :**
1. Quel modèle choisir ?
2. Comment interpréter les coefficients ?

#### Situation 3 : Banque
**Variable à prédire :** Décision de crédit
**Modalités :** {Refusé, Accordé avec conditions, Accordé sans conditions}
**Variables explicatives :** Revenu, Score crédit, Durée emploi

**Questions :**
1. Y a-t-il un ordre naturel ?
2. Quel modèle recommandez-vous ?

### Solutions

#### Solution 1 : E-commerce
```
Modèle : Multinomial logit
Justification : Les catégories sont nominales (pas d'ordre naturel)
Classe de référence : Choisir la plus fréquente (ex: Électronique)
```

#### Solution 2 : Hôpital  
```
Modèle : Régression ordinale (odds cumulatifs)
Justification : Ordre naturel clair de l'intensité
Interprétation : Un coefficient positif augmente la probabilité 
de douleurs plus intenses
```

#### Solution 3 : Banque
```
Ordre naturel : Oui (Refusé < Avec conditions < Sans conditions)
Modèle recommandé : Régression ordinale
Alternative : Si l'hypothèse de proportionnalité est violée, 
utiliser le modèle multinomial
```

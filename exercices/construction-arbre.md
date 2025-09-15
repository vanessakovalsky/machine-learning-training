# EXERCICE PRATIQUE : Construction d'arbre à la main (25min)

## Contexte
Vous êtes consultant pour une banque qui souhaite prédire l'octroi de crédit. Voici les données d'apprentissage :

| Client | Âge | Revenu | Historique | Crédit accordé |
|--------|-----|--------|------------|----------------|
| 1      | <30 | Faible | Bon        | Non            |
| 2      | <30 | Faible | Mauvais    | Non            |
| 3      | 30-40| Faible| Bon        | Oui            |
| 4      | >40 | Moyen  | Bon        | Oui            |
| 5      | >40 | Élevé  | Bon        | Oui            |
| 6      | >40 | Élevé  | Mauvais    | Non            |
| 7      | 30-40| Élevé | Mauvais    | Oui            |
| 8      | <30 | Moyen  | Bon        | Non            |
| 9      | <30 | Élevé  | Bon        | Oui            |
| 10     | >40 | Moyen  | Mauvais    | Oui            |
| 11     | <30 | Moyen  | Mauvais    | Oui            |
| 12     | 30-40| Moyen  | Bon       | Oui            |
| 13     | 30-40| Élevé  | Bon       | Oui            |
| 14     | >40 | Faible | Mauvais    | Non            |

## Étapes de l'exercice

### Étape 1 : Calcul de l'impureté initiale (5min)
1. Calculez l'impureté de Gini pour l'ensemble initial
2. Calculez l'entropie pour l'ensemble initial

### Étape 2 : Choix du premier nœud (10min)
Pour chaque attribut (Âge, Revenu, Historique) :
1. Calculez le gain d'information avec le critère de Gini
2. Déterminez le meilleur attribut pour la racine

### Étape 3 : Construction des branches (10min)
1. Créez les sous-arbres pour chaque valeur du meilleur attribut
2. Répétez le processus jusqu'aux feuilles
3. Dessinez l'arbre final

## Solution guidée

**Calculs initiaux :**
- Total : 14 exemples (9 Oui, 5 Non)
- Gini initial = 1 - (9/14)² - (5/14)² ≈ 0.459
- Entropie initiale = -(9/14)×log2(9/14) - (5/14)×log2(5/14) ≈ 0.940

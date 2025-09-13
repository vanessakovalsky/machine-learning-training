# Exercices pratiques - Module 2
*Préparation et exploration des données*

---

## 🛠️ Setup et imports nécessaires

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Imports pour le preprocessing
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE, mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")
```

---

## 📊 Exercice 1 : Nettoyage d'un dataset (25 minutes)

### Dataset : Ventes immobilières

```python
# Création d'un dataset synthétique avec des problèmes réalistes
np.random.seed(42)
n_samples = 1000

# Génération des données de base
data = {
    'surface': np.random.normal(120, 40, n_samples),
    'prix': np.random.normal(300000, 100000, n_samples),
    'nb_pieces': np.random.randint(1, 8, n_samples),
    'age_batiment': np.random.randint(0, 100, n_samples),
    'etage': np.random.randint(-1, 20, n_samples),  # -1 = sous-sol
    'balcon': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    'parking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'quartier': np.random.choice(['Centre', 'Nord', 'Sud', 'Est', 'Ouest'], n_samples),
    'type_chauffage': np.random.choice(['Gaz', 'Electrique', 'Fuel', 'Pompe_chaleur'], n_samples)
}

df_immobilier = pd.DataFrame(data)

# Ajout de corrélations réalistes
df_immobilier['prix'] = (df_immobilier['surface'] * 2500 + 
                        df_immobilier['nb_pieces'] * 15000 +
                        (100 - df_immobilier['age_batiment']) * 1000 +
                        df_immobilier['balcon'] * 20000 +
                        df_immobilier['parking'] * 25000 +
                        np.random.normal(0, 30000, n_samples))

# Introduction de valeurs manquantes de façon réaliste
# Les étages élevés ont plus de chances d'avoir des données manquantes
missing_prob = np.where(df_immobilier['etage'] > 10, 0.3, 0.1)
df_immobilier.loc[np.random.random(n_samples) < missing_prob, 'balcon'] = np.nan

# Données manquantes aléatoires
df_immobilier.loc[np.random.choice(df_immobilier.index, 50), 'type_chauffage'] = np.nan
df_immobilier.loc[np.random.choice(df_immobilier.index, 30), 'age_batiment'] = np.nan

# Introduction de valeurs aberrantes
outlier_indices = np.random.choice(df_immobilier.index, 20)
df_immobilier.loc[outlier_indices, 'surface'] = np.random.uniform(500, 1000, len(outlier_indices))
df_immobilier.loc[outlier_indices, 'prix'] = df_immobilier.loc[outlier_indices, 'surface'] * 3000

# Quelques erreurs de saisie
error_indices = np.random.choice(df_immobilier.index, 5)
df_immobilier.loc[error_indices, 'nb_pieces'] = 0  # 0 pièce = erreur
df_immobilier.loc[np.random.choice(df_immobilier.index, 3), 'etage'] = -10  # Étage impossible

print("Dataset créé avec succès!")
print(f"Forme du dataset: {df_immobilier.shape}")
df_immobilier.head()
```

### Tâches à réaliser

#### 1. Examen initial des données

```python
# TODO: Complétez le code suivant

print("=== INFORMATIONS GÉNÉRALES ===")
# Affichez les informations générales du dataset

print("\n=== STATISTIQUES DESCRIPTIVES ===")
# Affichez les statistiques descriptives

print("\n=== VALEURS MANQUANTES ===")
# Calculez et affichez le nombre et pourcentage de valeurs manquantes par colonne

print("\n=== VALEURS UNIQUES ===")
# Affichez le nombre de valeurs uniques par colonne
```

#### 2. Analyse des valeurs manquantes

```python
# TODO: Analysez les patterns de valeurs manquantes

# Visualisation des valeurs manquantes
import missingno as msno  # pip install missingno

# Créez une matrice de visualisation des valeurs manquantes
# Analysez les corrélations entre valeurs manquantes
# Proposez des hypothèses sur les mécanismes (MCAR, MAR, MNAR)
```

#### 3. Traitement des valeurs manquantes

```python
# Méthode 1: Imputation simple
def imputation_simple(df):
    """Imputation par moyenne/mode"""
    df_copy = df.copy()
    
    # TODO: Imputer les variables numériques par la moyenne
    # TODO: Imputer les variables catégorielles par le mode
    
    return df_copy

# Méthode 2: Imputation KNN
def imputation_knn(df):
    """Imputation par K-Nearest Neighbors"""
    df_copy = df.copy()
    
    # TODO: Préparer les données (encoder les variables catégorielles)
    # TODO: Appliquer KNNImputer
    # TODO: Remettre au format original
    
    return df_copy

# Testez les deux méthodes et comparez les résultats
```

#### 4. Détection des valeurs aberrantes

```python
# Méthode 1: Z-score
def detect_outliers_zscore(df, columns, threshold=3):
    """Détection par Z-score"""
    outliers = {}
    for col in columns:
        # TODO: Calculez les Z-scores
        # TODO: Identifiez les outliers (|z| > threshold)
        pass
    return outliers

# Méthode 2: IQR
def detect_outliers_iqr(df, columns):
    """Détection par méthode IQR"""
    outliers = {}
    for col in columns:
        # TODO: Calculez Q1, Q3 et IQR
        # TODO: Identifiez les outliers
        pass
    return outliers

# Visualisation des outliers
def plot_outliers(df, column):
    """Visualise les outliers avec box plot et histogram"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # TODO: Créez un box plot et un histogramme
    
    plt.tight_layout()
    plt.show()

# Testez sur les variables numériques
numeric_cols = ['surface', 'prix', 'age_batiment', 'etage']
for col in numeric_cols:
    plot_outliers(df_immobilier, col)
```

#### 5. Normalisation et standardisation

```python
def compare_scaling_methods(df, columns):
    """Compare différentes méthodes de mise à l'échelle"""
    
    # TODO: Appliquez StandardScaler, MinMaxScaler, RobustScaler
    # TODO: Créez des visualisations comparatives
    # TODO: Calculez les statistiques avant/après transformation
    
    fig, axes = plt.subplots(len(columns), 4, figsize=(16, len(columns)*4))
    
    for i, col in enumerate(columns):
        # Original
        axes[i,0].hist(df[col].dropna(), bins=30, alpha=0.7)
        axes[i,0].set_title(f'{col} - Original')
        
        # StandardScaler
        # TODO: Histogramme après standardisation
        
        # MinMaxScaler  
        # TODO: Histogramme après normalisation Min-Max
        
        # RobustScaler
        # TODO: Histogramme après normalisation robuste
    
    plt.tight_layout()
    plt.show()

# Testez la fonction
compare_scaling_methods(df_immobilier, numeric_cols)
```

#### 6. Choix et justification

```python
# TODO: Créez une fonction qui applique vos choix de prétraitement
def preprocess_data(df):
    """
    Fonction de prétraitement complète basée sur votre analyse
    
    Documentez vos choix:
    - Quelle méthode d'imputation et pourquoi?
    - Comment traiter les outliers et pourquoi?
    - Quelle normalisation et pourquoi?
    """
    df_processed = df.copy()
    
    # TODO: Appliquez vos transformations
    
    return df_processed

# Appliquez votre prétraitement
df_clean = preprocess_data(df_immobilier)

# Comparez avant/après
print("AVANT prétraitement:")
print(df_immobilier.isnull().sum())
print("\nAPRÈS prétraitement:")
print(df_clean.isnull().sum())
```

---

## 🔍 Exercice 2 : Analyse exploratoire (30 minutes)

### Dataset : Marketing Campaign

```python
# Création d'un dataset de campagne marketing
np.random.seed(123)
n_customers = 2000

# Génération des données clients
marketing_data = {
    'age': np.random.randint(18, 80, n_customers),
    'revenu_annuel': np.random.lognormal(10.5, 0.5, n_customers),
    'education': np.random.choice(['Lycee', 'Bachelor', 'Master', 'Doctorat'], 
                                 n_customers, p=[0.3, 0.4, 0.25, 0.05]),
    'situation_familiale': np.random.choice(['Celibataire', 'Marie', 'Divorce', 'Veuf'], 
                                          n_customers, p=[0.35, 0.45, 0.15, 0.05]),
    'nb_enfants': np.random.poisson(1.2, n_customers),
    'anciennete_client': np.random.randint(0, 20, n_customers),
    'nb_achats_web': np.random.poisson(5, n_customers),
    'nb_achats_magasin': np.random.poisson(3, n_customers),
    'nb_achats_catalogue': np.random.poisson(1, n_customers),
    'montant_vin': np.random.gamma(2, 100, n_customers),
    'montant_viande': np.random.gamma(2, 80, n_customers),
    'montant_poisson': np.random.gamma(1.5, 50, n_customers),
    'montant_sucre': np.random.gamma(1, 30, n_customers),
    'montant_or': np.random.gamma(0.5, 200, n_customers),
    'nb_visites_web': np.random.randint(0, 30, n_customers),
    'plainte_recente': np.random.choice([0, 1], n_customers, p=[0.9, 0.1])
}

df_marketing = pd.DataFrame(marketing_data)

# Ajout de corrélations réalistes
df_marketing['revenu_annuel'] = np.where(
    df_marketing['education'] == 'Doctorat', 
    df_marketing['revenu_annuel'] * 1.5,
    np.where(df_marketing['education'] == 'Master',
             df_marketing['revenu_annuel'] * 1.2,
             df_marketing['revenu_annuel'])
)

# La variable cible : acceptation de la campagne
# Plus probable si revenu élevé, client ancien, peu d'enfants
acceptance_prob = (
    (df_marketing['revenu_annuel'] / 100000) * 0.3 +
    (df_marketing['anciennete_client'] / 20) * 0.2 +
    (1 / (df_marketing['nb_enfants'] + 1)) * 0.3 +
    np.random.random(n_customers) * 0.2
)
df_marketing['accepte_campagne'] = (acceptance_prob > 0.5).astype(int)

print("Dataset marketing créé!")
print(f"Taux d'acceptation: {df_marketing['accepte_campagne'].mean():.2%}")
df_marketing.head()
```

### Tâches à réaliser

#### 1. Analyse univariée complète

```python
def analyse_univariee(df):
    """Analyse univariée complète du dataset"""
    
    # Séparation des variables
    numeric_vars = df.select_dtypes(include=[np.number]).columns
    categorical_vars = df.select_dtypes(include=['object']).columns
    
    print("=== VARIABLES NUMÉRIQUES ===")
    for var in numeric_vars:
        print(f"\n--- {var} ---")
        # TODO: Statistiques descriptives
        # TODO: Test de normalité (Shapiro-Wilk ou Kolmogorov-Smirnov)
        # TODO: Histogramme + courbe de densité
        # TODO: Box plot
    
    print("\n=== VARIABLES CATÉGORIELLES ===")
    for var in categorical_vars:
        print(f"\n--- {var} ---")
        # TODO: Table de fréquences (absolues et relatives)
        # TODO: Graphique en barres
        # TODO: Si pertinent, diagramme circulaire

# Exécutez l'analyse
analyse_univariee(df_marketing)
```

#### 2. Matrice de corrélation et interprétation

```python
def analyse_correlations(df):
    """Analyse des corrélations avec visualisations"""
    
    # Matrice de corrélation pour variables numériques
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    # TODO: Créez une heatmap de la matrice de corrélation
    # TODO: Identifiez les corrélations fortes (>0.7 ou <-0.7)
    # TODO: Créez des scatter plots pour les paires les plus corrélées
    
    # Analyse des corrélations avec la variable cible
    if 'accepte_campagne' in df.columns:
        target_corr = corr_matrix['accepte_campagne'].abs().sort_values(ascending=False)
        print("Corrélations avec la variable cible:")
        print(target_corr)
    
    return corr_matrix

corr_matrix = analyse_correlations(df_marketing)
```

#### 3. Détection de la multicolinéarité

```python
def detect_multicollinearity(df):
    """Détecte la multicolinéarité avec VIF"""
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    # TODO: Calculez le VIF pour chaque variable
    # TODO: Identifiez les variables problématiques (VIF > 5)
    # TODO: Proposez des solutions (suppression, transformation, etc.)
    
    # Matrice de corrélation avec seuil
    corr_matrix = numeric_df.corr().abs()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print("Paires de variables fortement corrélées (>0.8):")
        for var1, var2, corr in high_corr_pairs:
            print(f"{var1} - {var2}: {corr:.3f}")
    
    return high_corr_pairs

multicollinear_vars = detect_multicollinearity(df_marketing)
```

#### 4. Techniques de sélection de features

```python
def feature_selection_comparison(X, y):
    """Compare plusieurs méthodes de sélection de features"""
    
    results = {}
    
    # Méthode 1: Corrélation avec la cible
    corr_scores = X.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
    results['correlation'] = corr_scores.head(10).index.tolist()
    
    # Méthode 2: Chi-2 (pour variables numériques, on peut discrétiser)
    # TODO: Appliquez SelectKBest avec chi2
    
    # Méthode 3: Mutual Information
    # TODO: Utilisez mutual_info_classif
    
    # Méthode 4: Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    results['random_forest'] = rf_importance.sort_values(ascending=False).head(10).index.tolist()
    
    # Méthode 5: RFE avec Random Forest
    rfe = RFE(rf, n_features_to_select=10)
    rfe.fit(X, y)
    results['rfe'] = X.columns[rfe.support_].tolist()
    
    # Comparaison des résultats
    print("=== COMPARAISON DES MÉTHODES DE SÉLECTION ===")
    for method, features in results.items():
        print(f"\n{method.upper()}:")
        print(features)
    
    # Visualisation des importances
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Corrélation
    axes[0,0].barh(range(len(corr_scores.head(10))), corr_scores.head(10).values)
    axes[0,0].set_yticks(range(len(corr_scores.head(10))))
    axes[0,0].set_yticklabels(corr_scores.head(10).index)
    axes[0,0].set_title('Top 10 - Corrélation avec la cible')
    
    # Random Forest Importance
    axes[0,1].barh(range(len(rf_importance.head(10))), rf_importance.head(10).values)
    axes[0,1].set_yticks(range(len(rf_importance.head(10))))
    axes[0,1].set_yticklabels(rf_importance.head(10).index)
    axes[0,1].set_title('Top 10 - Random Forest Importance')
    
    # TODO: Ajoutez les visualisations pour Chi-2 et Mutual Information
    
    plt.tight_layout()
    plt.show()
    
    return results

# Préparation des données pour la sélection
X = df_marketing.drop('accepte_campagne', axis=1)
y = df_marketing['accepte_campagne']

# Encodage des variables catégorielles pour certaines méthodes
X_encoded = pd.get_dummies(X, drop_first=True)

# Exécution de la comparaison
feature_results = feature_selection_comparison(X_encoded, y)
```

#### 5. Analyse bivariée approfondie

```python
def analyse_bivariee_complete(df, target_col):
    """Analyse bivariée complète avec la variable cible"""
    
    numeric_vars = df.select_dtypes(include=[np.number]).columns
    categorical_vars = df.select_dtypes(include=['object']).columns
    
    print("=== ANALYSE BIVARIÉE AVEC LA VARIABLE CIBLE ===")
    
    # Variables numériques vs cible
    for var in numeric_vars:
        if var != target_col:
            print(f"\n--- {var} vs {target_col} ---")
            
            # TODO: Box plot par groupe de la cible
            plt.figure(figsize=(10, 6))
            df.boxplot(column=var, by=target_col)
            plt.title(f'{var} par {target_col}')
            plt.show()
            
            # TODO: Test statistique (t-test ou Mann-Whitney)
            group_0 = df[df[target_col] == 0][var].dropna()
            group_1 = df[df[target_col] == 1][var].dropna()
            
            # Test de normalité
            _, p_norm_0 = stats.shapiro(group_0.sample(min(5000, len(group_0))))
            _, p_norm_1 = stats.shapiro(group_1.sample(min(5000, len(group_1))))
            
            if p_norm_0 > 0.05 and p_norm_1 > 0.05:
                # T-test si les données sont normales
                stat, p_value = stats.ttest_ind(group_0, group_1)
                test_name = "T-test"
            else:
                # Mann-Whitney sinon
                stat, p_value = stats.mannwhitneyu(group_0, group_1)
                test_name = "Mann-Whitney U"
            
            print(f"{test_name}: statistic={stat:.4f}, p-value={p_value:.4f}")
    
    # Variables catégorielles vs cible
    for var in categorical_vars:
        print(f"\n--- {var} vs {target_col} ---")
        
        # TODO: Table de contingence
        contingency_table = pd.crosstab(df[var], df[target_col])
        print("Table de contingence:")
        print(contingency_table)
        
        # TODO: Test du Chi-2
        chi2_stat, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Test Chi-2: statistic={chi2_stat:.4f}, p-value={p_chi2:.4f}")
        
        # TODO: Graphique en barres empilées ou groupées
        plt.figure(figsize=(10, 6))
        contingency_table.plot(kind='bar', stacked=True)
        plt.title(f'{var} vs {target_col}')
        plt.xticks(rotation=45)
        plt.show()

# Exécution de l'analyse bivariée
analyse_bivariee_complete(df_marketing, 'accepte_campagne')
```

#### 6. Rapport de synthèse et recommandations

```python
def generate_analysis_report(df, target_col, feature_results, multicollinear_vars):
    """Génère un rapport de synthèse de l'analyse exploratoire"""
    
    print("=" * 60)
    print("         RAPPORT DE SYNTHÈSE - ANALYSE EXPLORATOIRE")
    print("=" * 60)
    
    # 1. Aperçu général des données
    print(f"\n1. APERÇU GÉNÉRAL")
    print(f"   - Nombre d'observations: {len(df):,}")
    print(f"   - Nombre de variables: {len(df.columns)}")
    print(f"   - Taux de la classe positive ({target_col}): {df[target_col].mean():.2%}")
    
    # 2. Qualité des données
    missing_percent = (df.isnull().sum() / len(df) * 100)
    vars_with_missing = missing_percent[missing_percent > 0]
    
    print(f"\n2. QUALITÉ DES DONNÉES")
    if len(vars_with_missing) > 0:
        print(f"   - Variables avec données manquantes: {len(vars_with_missing)}")
        for var, pct in vars_with_missing.items():
            print(f"     • {var}: {pct:.1f}%")
    else:
        print("   - Aucune donnée manquante détectée")
    
    # 3. Multicolinéarité
    print(f"\n3. MULTICOLINÉARITÉ")
    if len(multicollinear_vars) > 0:
        print(f"   - {len(multicollinear_vars)} paires de variables fortement corrélées:")
        for var1, var2, corr in multicollinear_vars:
            print(f"     • {var1} ↔ {var2}: {corr:.3f}")
    else:
        print("   - Aucun problème de multicolinéarité majeur détecté")
    
    # 4. Variables les plus importantes
    print(f"\n4. VARIABLES LES PLUS IMPORTANTES")
    print("   Consensus entre les différentes méthodes:")
    
    # Compter les occurrences de chaque variable
    all_features = []
    for features in feature_results.values():
        all_features.extend(features)
    
    feature_counts = pd.Series(all_features).value_counts()
    top_features = feature_counts.head(10)
    
    for feature, count in top_features.items():
        print(f"     • {feature}: mentionnée dans {count}/{len(feature_results)} méthodes")
    
    # 5. Recommandations
    print(f"\n5. RECOMMANDATIONS")
    print("   a) Variables à conserver en priorité:")
    recommended_features = top_features[top_features >= len(feature_results)//2].index.tolist()
    for feature in recommended_features[:10]:
        print(f"      • {feature}")
    
    print("\n   b) Variables à examiner pour multicolinéarité:")
    if multicollinear_vars:
        multicollinear_features = set()
        for var1, var2, corr in multicollinear_vars:
            multicollinear_features.update([var1, var2])
        for feature in multicollinear_features:
            print(f"      • {feature}")
    
    print("\n   c) Actions de prétraitement recommandées:")
    numeric_vars = df.select_dtypes(include=[np.number]).columns
    
    # Test de normalité pour recommander la transformation
    non_normal_vars = []
    for var in numeric_vars:
        if var != target_col:
            _, p_value = stats.normaltest(df[var].dropna())
            if p_value < 0.05:
                non_normal_vars.append(var)
    
    if non_normal_vars:
        print(f"      • Transformation recommandée pour: {', '.join(non_normal_vars[:5])}")
    
    print("      • Standardisation recommandée pour tous les modèles basés sur la distance")
    
    return {
        'recommended_features': recommended_features,
        'multicollinear_features': list(multicollinear_features) if multicollinear_vars else [],
        'non_normal_vars': non_normal_vars
    }

# Génération du rapport
report_results = generate_analysis_report(df_marketing, 'accepte_campagne', 
                                        feature_results, multicollinear_vars)
```

---

## 📋 Solutions et corrections

### Correction Exercice 1

```python
# Solution complète pour l'exercice 1
def solution_exercice_1():
    """Solution complète de l'exercice de nettoyage"""
    
    # 1. Examen initial
    print("=== INFORMATIONS GÉNÉRALES ===")
    print(df_immobilier.info())
    print(f"Forme du dataset: {df_immobilier.shape}")
    
    print("\n=== STATISTIQUES DESCRIPTIVES ===")
    print(df_immobilier.describe())
    
    print("\n=== VALEURS MANQUANTES ===")
    missing_data = pd.DataFrame({
        'Nombre': df_immobilier.isnull().sum(),
        'Pourcentage': df_immobilier.isnull().sum() / len(df_immobilier) * 100
    })
    print(missing_data[missing_data['Nombre'] > 0])
    
    # 2. Imputation optimisée
    def imputation_optimisee(df):
        df_clean = df.copy()
        
        # Balcon: imputation logique (si pas de balcon, alors 0)
        df_clean['balcon'] = df_clean['balcon'].fillna(0)
        
        # Type chauffage: imputation par le mode du quartier
        df_clean['type_chauffage'] = df_clean.groupby('quartier')['type_chauffage'].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Gaz')
        )
        
        # Âge bâtiment: imputation par la médiane
        df_clean['age_batiment'] = df_clean['age_batiment'].fillna(
            df_clean['age_batiment'].median()
        )
        
        return df_clean
    
    # 3. Traitement des outliers
    def traiter_outliers(df):
        df_clean = df.copy()
        
        # Correction des erreurs évidentes
        df_clean.loc[df_clean['nb_pieces'] == 0, 'nb_pieces'] = 1
        df_clean.loc[df_clean['etage'] < -1, 'etage'] = 0
        
        # Winsorisation pour les surfaces extrêmes
        q99 = df_clean['surface'].quantile(0.99)
        q01 = df_clean['surface'].quantile(0.01)
        df_clean['surface'] = df_clean['surface'].clip(lower=q01, upper=q99)
        
        return df_clean
    
    # Application du pipeline complet
    df_final = imputation_optimisee(df_immobilier)
    df_final = traiter_outliers(df_final)
    
    # Normalisation
    numeric_cols = ['surface', 'prix', 'age_batiment', 'etage']
    scaler = RobustScaler()  # Choix du RobustScaler car moins sensible aux outliers
    df_final[numeric_cols] = scaler.fit_transform(df_final[numeric_cols])
    
    print("\n=== RÉSULTATS FINAUX ===")
    print(f"Données manquantes après traitement: {df_final.isnull().sum().sum()}")
    print(f"Forme finale: {df_final.shape}")
    
    return df_final

# Exécution de la solution
df_solution = solution_exercice_1()
```

### Correction Exercice 2

```python
# Solution pour la sélection de features optimale
def solution_feature_selection(df, target_col):
    """Sélection optimale des features basée sur l'analyse complète"""
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encodage des variables catégorielles
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Étape 1: Élimination des features à faible variance
    from sklearn.feature_selection import VarianceThreshold
    selector_var = VarianceThreshold(threshold=0.01)
    X_var = selector_var.fit_transform(X_encoded)
    features_var = X_encoded.columns[selector_var.get_support()]
    print(f"Après élimination variance: {len(features_var)} features")
    
    # Étape 2: Élimination des features corrélées
    X_var_df = pd.DataFrame(X_var, columns=features_var)
    corr_matrix = X_var_df.corr().abs()
    
    # Trouver les paires corrélées
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Identifier les features à supprimer
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > 0.9)]
    
    X_uncorr = X_var_df.drop(columns=to_drop)
    print(f"Après élimination corrélation: {len(X_uncorr.columns)} features")
    
    # Étape 3: Sélection finale par importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_uncorr, y)
    
    importance_scores = pd.DataFrame({
        'feature': X_uncorr.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Sélection des top 15 features
    final_features = importance_scores.head(15)['feature'].tolist()
    
    print("\n=== FEATURES FINALES SÉLECTIONNÉES ===")
    for i, (_, row) in enumerate(importance_scores.head(15).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:25} {row['importance']:.4f}")
    
    return final_features, importance_scores

# Application de la sélection optimale
final_features, feature_importance = solution_feature_selection(df_marketing, 'accepte_campagne')
```

---

## ✅ Checklist de validation

### Exercice 1 - Nettoyage de données
- [ ] Identification correcte des types de données manquantes
- [ ] Application d'au moins 2 méthodes d'imputation différentes
- [ ] Détection des outliers avec 2 techniques minimum
- [ ] Justification des choix de traitement des outliers
- [ ] Application et comparaison de 3 méthodes de normalisation
- [ ] Documentation des transformations appliquées

### Exercice 2 - Exploration et sélection
- [ ] Analyse univariée complète (statistiques + visualisations)
- [ ] Matrice de corrélation avec interprétation
- [ ] Détection de la multicolinéarité (VIF ou corrélation)
- [ ] Application de 3 techniques de sélection de features minimum
- [ ] Comparaison des résultats des différentes méthodes
- [ ] Recommandations finales justifiées avec rapport de synthèse

---

## 💡 Points clés à retenir

### Prétraitement
- **Toujours analyser avant de traiter** : comprendre les mécanismes des données manquantes
- **Outliers ≠ erreurs** : distinguer les valeurs aberrantes des erreurs de mesure
- **Choisir la normalisation selon la distribution** et l'algorithme cible
- **Documenter toutes les transformations** pour la reproductibilité

### Exploration
- **L'EDA guide le preprocessing** : les insights découverts orientent les choix
- **Multicolinéarité** : peut affecter l'interprétabilité sans nuire à la prédiction
- **Combiner plusieurs approches** de sélection pour une sélection robuste
- **Valider sur des données test** : éviter l'overfitting dans la sélection

### Workflow recommandé
1. **Exploration initiale** → Comprendre les données
2. **Nettoyage** → Traiter les problèmes identifiés  
3. **Transformation** → Préparer pour le modeling
4. **Sélection** → Identifier les features pertinentes
5. **Validation** → Vérifier la qualité des transformations

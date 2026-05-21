# Machine Learning Analysis – Adult Income Dataset (UCI)

**Auteur : Erwan Blouin**

## Objectif

L’objectif de ce projet est de prédire si un individu gagne plus ou moins de **50K$ par an** à partir de données démographiques, sociales et économiques.

Le projet compare les performances de trois algorithmes de Machine Learning :

- Decision Tree
- Random Forest
- MLP Classifier

## Dataset

| Propriété | Valeur |
|---|---|
| Source | UCI Machine Learning Repository |
| Taille | 48 842 individus |
| Variable cible | `income` : `<=50K` ou `>50K` |
| Type de problème | Classification binaire supervisée |

## Variables principales

| Variable | Type | Description |
|---|---|---|
| `age` | Numérique | Âge de l'individu |
| `workclass` | Catégorielle | Type d'employeur |
| `education` | Catégorielle | Niveau d'éducation |
| `education-num` | Numérique | Représentation numérique du niveau d'éducation |
| `marital-status` | Catégorielle | Situation matrimoniale |
| `occupation` | Catégorielle | Type de poste |
| `hours-per-week` | Numérique | Heures travaillées par semaine |
| `capital-gain` / `capital-loss` | Numérique | Gains et pertes en capital |
| `native-country` | Catégorielle | Pays d'origine |

## Pipeline de prétraitement

Les données brutes sont d’abord nettoyées puis transformées avant l’entraînement des modèles.

Étapes principales :

1. Suppression des valeurs manquantes représentées par `?` ou `NaN`
2. Séparation des variables numériques et catégorielles
3. Application d’un `ColumnTransformer`
4. Standardisation des colonnes numériques avec `StandardScaler`
5. Encodage des colonnes catégorielles avec `OneHotEncoder`
6. Création d’un pipeline complet avec le prétraitement et le modèle

Le dataset est ensuite séparé en données d’entraînement et données de test :

```python
Train / Test split : 80% / 20%
random_state = 42
```

## Modèles entraînés

| Modèle | Caractéristiques |
|---|---|
| Decision Tree | Baseline simple, interprétable, mais peut facilement overfitter |
| Random Forest | Ensemble d’arbres de décision, plus stable et plus robuste |
| MLP Classifier | Réseau de neurones capable de capturer des relations non linéaires |

## Métriques d’évaluation

Le dataset est déséquilibré, avec environ **75% des individus gagnant <=50K** et **25% gagnant >50K**.

Pour cette raison, le **F1-score** est utilisé comme métrique principale, car il est plus fiable que l’accuracy seule dans le cas d’un dataset déséquilibré.

Les métriques utilisées sont :

| Métrique | Description |
|---|---|
| Accuracy | Taux global de bonnes prédictions |
| Precision | Qualité des prédictions positives |
| Recall | Capacité du modèle à détecter les hauts revenus |
| F1-score | Moyenne harmonique entre la precision et le recall |

## Résultats

| Modèle | Accuracy | F1-score pour `>50K` |
|---|---:|---:|
| Decision Tree | ~85% | ~65% |
| Random Forest | ~86% | ~70% |
| MLP Classifier | ~87% | ~72% |

Le **MLP Classifier** obtient les meilleures performances globales grâce à la combinaison du scaling des variables numériques et de l’encodage one-hot des variables catégorielles.

## Conclusions

Les facteurs les plus influents sur le revenu sont notamment :

- le niveau d’éducation
- le nombre d’heures travaillées par semaine
- le type de poste occupé
- la situation matrimoniale
- les gains en capital

Le **Random Forest** offre le meilleur compromis entre robustesse, performance et interprétabilité.

Le **MLP Classifier** obtient les meilleures métriques, mais il est plus coûteux en temps de calcul.

Le **Decision Tree** reste utile comme modèle de base simple et interprétable.

## Stack technique

Le projet utilise les bibliothèques Python suivantes :

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

Modules principaux utilisés avec scikit-learn :

- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `MLPClassifier`
- `Pipeline`
- `ColumnTransformer`
- `StandardScaler`
- `OneHotEncoder`
- `classification_report`
- `confusion_matrix`

## Lancer le projet

### 1. Cloner le repository

```bash
git clone https://github.com/erwan-blouin/adult-income-ml
```

### 2. Installer les dépendances

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 3. Lancer le notebook

```bash
jupyter notebook Jupyter_Notebook.ipynb
```

## Remarque

Les fichiers `adult.data` et `adult.test` doivent être placés dans le même répertoire que le notebook.

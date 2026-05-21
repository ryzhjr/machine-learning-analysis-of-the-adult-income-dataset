 Machine Learning Analysis – Adult Income Dataset (UCI)
Erwan Blouin 


 Objectif
Prédire si un individu gagne plus ou moins de 50K$/an à partir de données démographiques, sociales et économiques, en comparant les performances de trois algorithmes de Machine Learning.

Dataset
PropriétéValeurSourceUCI Machine Learning RepositoryTaille48 842 individusVariable cibleincome → <=50K ou >50KType de problèmeClassification binaire supervisée
Variables principales
VariableTypeDescriptionageNumériqueÂge de l'individuworkclassCatégorielleType d'employeureducationCatégorielleNiveau d'éducationeducation-numNumériqueReprésentation numérique du niveau d'éducationmarital-statusCatégorielleSituation matrimonialeoccupationCatégorielleType de postehours-per-weekNumériqueHeures travaillées par semainecapital-gain/lossNumériqueGains/pertes en capitalnative-countryCatégoriellePays d'origine

 Pipeline de prétraitement
Raw Data
   │
   ├── Suppression des valeurs manquantes ("?" et NaN)
   │
   ├── Séparation numérique / catégorielle
   │
   ├── ColumnTransformer
   │     ├── StandardScaler      → colonnes numériques
   │     └── OneHotEncoder       → colonnes catégorielles
   │
   └── Pipeline(preprocessor + model)
Train / Test split : 80% / 20% — random_state=42

 Modèles entraînés
ModèleCaractéristiquesDecision TreeBaseline simple, interprétable, tend à overfitterRandom ForestEnsemble de arbres, plus stable et robusteMLP ClassifierRéseau de neurones, capture les interactions non-linéaires

 Métriques d'évaluation

 Le dataset est déséquilibré (~75% <=50K / ~25% >50K). Le F1-score est la métrique principale, plus fiable que l'accuracy seule.


Accuracy – Taux de bonnes prédictions global
Precision – Qualité des prédictions positives
Recall – Capacité à détecter les hauts revenus
F1-score – Moyenne harmonique precision/recall


 Résultats
ModèleAccuracyF1-score (>50K)Decision Tree~85%~65%Random Forest~86%~70%MLP Classifier~87%~72%

Le MLP Classifier obtient les meilleures performances globales grâce au scaling et à l'encodage one-hot.


 Conclusions

Les facteurs les plus influents sur le revenu sont : niveau d'éducation, heures travaillées, occupation, situation matrimoniale et capital gain
Le Random Forest offre le meilleur compromis robustesse / interprétabilité
Le MLP atteint les meilleures métriques mais est plus coûteux en calcul
Le Decision Tree reste utile comme baseline interprétable


 Stack technique
Python 3.x
├── pandas
├── numpy
├── matplotlib
└── scikit-learn
      ├── DecisionTreeClassifier
      ├── RandomForestClassifier
      ├── MLPClassifier
      ├── Pipeline / ColumnTransformer
      ├── StandardScaler / OneHotEncoder
      └── classification_report, confusion_matrix

 Lancer le projet
 Cloner le repo
git clone https://github.com/erwan-blouin/adult-income-ml

# Installer les dépendances
pip install pandas numpy matplotlib scikit-learn

# Lancer le notebook
jupyter notebook Jupyter_Notebook.ipynb

 Placer les fichiers adult.data et adult.test dans le même répertoire que le notebook.

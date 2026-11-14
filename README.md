# Prédiction du risque de santé mentale chez les étudiants

Projet de machine learning visant à prédire le niveau de risque de santé mentale
(`Low`, `Medium` ou `High`) d'un étudiant à partir de caractéristiques
académiques, comportementales, sociales et de mode de vie.

> **Avertissement :** ce projet est éducatif. Le dataset est synthétique et les
> prédictions ne doivent pas être utilisées pour établir un diagnostic, effectuer
> un dépistage clinique, prendre une décision disciplinaire ou sélectionner des
> étudiants.

## Sommaire

- [Objectifs](#objectifs)
- [Dataset](#dataset)
- [Démarche](#démarche)
- [Résultats](#résultats)
- [Structure du projet](#structure-du-projet)
- [Installation et utilisation](#installation-et-utilisation)
- [Orchestration Airflow](#orchestration-airflow)

## Objectifs

Le projet compare trois modèles de classification multi-classe :

- **Régression logistique** : baseline interprétable ;
- **Random Forest** : modèle non linéaire adapté aux variables mixtes ;
- **XGBoost** : modèle de boosting pour données tabulaires.

L'objectif est de comparer leurs performances avec une validation croisée
stratifiée, puis d'optimiser leurs hyperparamètres. La métrique principalement
retenue est le **F1-score macro**, car elle donne le même poids aux classes
`Low`, `Medium` et `High`.

## Dataset

Le fichier attendu est `data/raw/student_mental_health.csv`. Il provient du
dataset Kaggle
[`maqdadali/mental-health-risk-prediction-dataset-students`](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students).
La description détaillée des variables se trouve dans
[`data/raw/README.md`](data/raw/README.md).

| Propriété | Valeur |
|---|---:|
| Observations | 2 000 |
| Colonnes | 17 |
| Variables explicatives | 15 après retrait de `Student_ID` |
| Cible | `Mental_Health_Risk` |
| Classes | `Low`, `Medium`, `High` |
| Répartition des classes | `Low`: 660, `Medium`: 660, `High`: 680 |
| Valeurs manquantes | 419 au total, dans 7 colonnes |

Les variables comprennent notamment l'âge, le genre, l'année d'étude, le sommeil,
le temps d'étude, le GPA, le temps d'écran et de réseaux sociaux, l'activité
physique, le stress financier, le soutien familial, l'assiduité, le régime
alimentaire et les antécédents de santé mentale. Le dataset est synthétique :
il ne décrit pas de personnes réelles.

## Démarche

1. **Chargement** : vérification de la présence et de la lisibilité du CSV.
2. **Séparation des données** : retrait de `Student_ID` et de la cible des
   variables explicatives, puis séparation stratifiée 80/20
   (`random_state=42`).
3. **Prétraitement**, ajusté uniquement sur l'ensemble d'entraînement :
   - imputation médiane des variables numériques ;
   - standardisation des variables numériques ;
   - imputation par la modalité la plus fréquente des variables catégorielles ;
   - encodage one-hot des variables catégorielles ;
   - encodage ordinal de la cible.
4. **Validation** : validation croisée stratifiée en 5 folds sur l'entraînement,
   avec calcul de l'accuracy, de la précision macro, du rappel macro et du
   F1-score macro.
5. **Entraînement et test** : ajustement des modèles sur l'entraînement et
   évaluation finale sur le jeu de test conservé.
6. **Optimisation** : recherche d'hyperparamètres pour chaque modèle ; les
   résultats sont conservés dans
   `results/hyperparameter_optimization_summary.csv`.

Les fonctions réutilisables sont regroupées dans [`src/`](src/), le pipeline
Airflow dans [`airflow/dags/mental_health_risk_pipeline.py`](airflow/dags/mental_health_risk_pipeline.py)
et les expérimentations dans [`notebooks/`](notebooks/).

## Résultats

### Baselines — validation croisée sur l'entraînement

| Modèle | Accuracy | Précision macro | Rappel macro | F1 macro |
|---|---:|---:|---:|---:|
| Régression logistique | 0,7500 | 0,7494 | 0,7494 | **0,7486** |
| XGBoost | 0,6950 | 0,6960 | 0,6942 | 0,6942 |
| Random Forest | 0,6794 | 0,6715 | 0,6783 | 0,6725 |

### Modèles optimisés — évaluation sur le jeu de test

| Modèle | F1 macro CV | Accuracy test | Précision macro test | Rappel macro test | F1 macro test |
|---|---:|---:|---:|---:|---:|
| Régression logistique | **0,7507** | **0,7525** | **0,7519** | **0,7518** | **0,7518** |
| XGBoost | 0,7033 | 0,7325 | 0,7259 | 0,7316 | 0,7275 |
| Random Forest | 0,6817 | 0,7125 | 0,7087 | 0,7118 | 0,7100 |

La régression logistique optimisée est le meilleur modèle selon le F1 macro sur
le test. Les matrices de confusion et les tableaux complets sont disponibles
dans [`results/`](results/), notamment
[`results/optimized_model_test_results.csv`](results/optimized_model_test_results.csv).
Ces scores mesurent la performance sur ce dataset synthétique et ne constituent
pas une validation clinique.

## Structure du projet

```text
.
├── airflow/dags/              # DAG de traitement et d'entraînement
├── app/                       # Application Streamlit de démonstration
├── data/raw/                  # Dataset source (à télécharger)
├── data/processed/            # Splits et artefacts de prétraitement
├── models/                    # Modèles entraînés
├── notebooks/                 # EDA, prétraitement, entraînement, optimisation
├── results/                   # Métriques, recherches et matrices de confusion
├── scripts/fetchdata.py       # Téléchargement du dataset Kaggle
├── src/                       # Code Python réutilisable
└── requirements.txt           # Dépendances Python
```

## Installation et utilisation

Le projet utilise Python 3.12 dans l'environnement de développement fourni.

```bash
git clone https://github.com/Moussa-Mohammed1/mental-health-risk-prediction.git
cd mental-health-risk-prediction
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Télécharger le dataset

Le fichier CSV est ignoré par Git. Téléchargez-le avec `kagglehub` :

```bash
python scripts/fetchdata.py
```

Une authentification Kaggle peut être nécessaire selon la configuration de
`kagglehub`. Le fichier doit ensuite être présent à
`data/raw/student_mental_health.csv`.

### Lancer la démonstration Streamlit

Après avoir téléchargé les données et généré les artefacts nécessaires :

```bash
streamlit run app/streamlit_app.py
```

L'application affiche les métriques du modèle optimisé et permet de tester une
prédiction technique à partir d'un formulaire. Elle ne remplace pas un
professionnel de santé.

## Orchestration Airflow

Le DAG [`mental_health_risk_pipeline`](airflow/dags/mental_health_risk_pipeline.py)
enchaîne trois tâches :

1. `prepare_data` vérifie que le dataset peut être chargé ;
2. `preprocess_data` crée les splits stratifiés et sauvegarde les artefacts ;
3. `train_and_evaluate` exécute la validation croisée, entraîne les modèles,
   évalue le jeu de test et écrit les résultats.

Le DAG est planifié avec `@weekly` et ne rattrape pas les exécutions historiques
(`catchup=False`).

```bash
export AIRFLOW_HOME="$PWD/airflow"
airflow db migrate
airflow dags list
airflow dags test mental_health_risk_pipeline 2026-11-09
airflow dags trigger mental_health_risk_pipeline
```

Pour exécuter le DAG, le fichier
`data/raw/student_mental_health.csv` doit être disponible depuis la racine du
projet. Les sorties sont écrites dans `data/processed/`, `models/` et
`results/`.

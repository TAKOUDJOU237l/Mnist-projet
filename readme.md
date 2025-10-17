#  Classification de Chiffres MNIST avec Deep Learning et MLflow

##  Description du projet

Ce projet implémente un réseau de neurones profond (Deep Neural Network) pour la classification de chiffres manuscrits en utilisant le dataset MNIST. Le projet intègre **MLflow** pour le tracking complet des expérimentations, permettant une gestion professionnelle du cycle de vie du modèle de Machine Learning.

## Objectifs

- Construire un modèle de classification performant pour reconnaître les chiffres de 0 à 9
- Implémenter le tracking des expériences avec MLflow
- Gérer le versioning des modèles et des hyperparamètres
- Faciliter la reproductibilité et la comparaison des expérimentations

##  Dataset : MNIST

Le dataset MNIST (Modified National Institute of Standards and Technology) est une référence en Deep Learning :

- **60 000 images** d'entraînement
- **10 000 images** de test
- Images en **niveaux de gris** de **28×28 pixels**
- **10 classes** (chiffres de 0 à 9)
- Dataset préchargé via `keras.datasets.mnist`

## Architecture du Modèle

### Structure du réseau

Le modèle utilise une architecture de réseau de neurones entièrement connecté (Fully Connected Network) :

```
Input (784) → Dense (512, ReLU) → Dropout (0.2) → Dense (10, Softmax)
```

**Détails des couches :**

1. **Couche d'entrée** : 784 neurones (28×28 pixels aplatis)
2. **Couche cachée Dense** : 512 neurones avec activation ReLU
3. **Couche Dropout** : 20% de dropout pour la régularisation
4. **Couche de sortie** : 10 neurones avec activation Softmax (probabilités par classe)

**Nombre total de paramètres entraînables** : ~407 050 paramètres

### Choix techniques

- **Activation ReLU** : Fonction d'activation non-linéaire efficace pour les couches cachées
- **Dropout** : Prévient le surapprentissage en désactivant aléatoirement des neurones
- **Softmax** : Transforme les sorties en distribution de probabilité pour la classification multi-classes
- **Optimiseur Adam** : Convergence rapide avec adaptation automatique du learning rate

## ⚙️ Hyperparamètres

| Hyperparamètre | Valeur | Description |
|----------------|--------|-------------|
| **Epochs** | 5 | Nombre de passages complets sur le dataset |
| **Batch Size** | 128 | Nombre d'exemples par batch |
| **Validation Split** | 0.1 (10%) | Proportion des données d'entraînement pour la validation |
| **Hidden Units** | 512 | Nombre de neurones dans la couche cachée |
| **Dropout Rate** | 0.2 (20%) | Taux de dropout pour la régularisation |
| **Optimizer** | Adam | Algorithme d'optimisation |
| **Loss Function** | Sparse Categorical Crossentropy | Fonction de perte pour classification |

##  Installation et Configuration

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
pip install tensorflow numpy mlflow
```

Ou avec un fichier `requirements.txt` :

```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt` :**
```
tensorflow==2.15.0
numpy==1.24.3
mlflow==2.8.0
```

## Structure du Projet

```
MNIST_MLflow_Project/
├── train_model.py          # Script principal d'entraînement
├── mnist_model.h5          # Modèle entraîné (généré)
├── requirements.txt        # Dépendances Python
├── README.md               # Documentation
└── mlruns/                 # Dossier MLflow (généré automatiquement)
    └── 0/
        └── [run_id]/
            ├── artifacts/  # Modèles et fichiers sauvegardés
            ├── metrics/    # Métriques enregistrées
            ├── params/     # Hyperparamètres
            └── tags/       # Tags et métadonnées
```

## Utilisation

### 1. Entraîner le modèle

Lancez le script d'entraînement :

```bash
python train_model.py
```

**Ce que fait le script :**
1.  Charge automatiquement le dataset MNIST
2.  Prétraite les données (normalisation et redimensionnement)
3.  Construit le modèle de réseau de neurones
4.  Entraîne le modèle sur 5 époques
5.  Évalue les performances sur le jeu de test
6.  Enregistre toutes les métriques et paramètres dans MLflow
7.  Sauvegarde le modèle au format `.h5` et dans le Model Registry MLflow

### 2. Visualiser les résultats avec MLflow UI

Après l'entraînement, lancez l'interface web MLflow :

```bash
mlflow ui
```

Puis ouvrez votre navigateur à l'adresse : **http://localhost:5000**

### 3. Explorer l'interface MLflow

Dans MLflow UI, vous pouvez :

-  **Comparer différentes exécutions** (runs) côte à côte
-  **Visualiser l'évolution des métriques** (accuracy, loss) par époque
-  **Filtrer et trier** les expériences par performances
-  **Télécharger les modèles** sauvegardés
-  **Gérer les versions** dans le Model Registry

##  Métriques Trackées

### Hyperparamètres enregistrés
- Nombre d'époques, batch size, validation split
- Architecture du réseau (hidden units, dropout rate)
- Optimiseur et fonctions d'activation
- Informations sur le dataset (nombre d'échantillons, classes)

### Métriques d'entraînement (par époque)
- `train_loss` : Perte sur les données d'entraînement
- `train_accuracy` : Précision sur les données d'entraînement
- `val_loss` : Perte sur les données de validation
- `val_accuracy` : Précision sur les données de validation

### Métriques finales
- `test_loss` : Perte sur le jeu de test
- `test_accuracy` : Précision finale sur le jeu de test

## Résultats Attendus

Le modèle atteint généralement une **précision de ~97-98%** sur le jeu de test après 5 époques d'entraînement.

**Exemple de sortie :**
```
Précision sur les données de test : 0.9785
Perte sur les données de test : 0.0721
```

##  Workflow du Projet

```
1. Chargement des données MNIST
         ↓
2. Prétraitement (normalisation + reshape)
         ↓
3. Construction du modèle
         ↓
4. Compilation avec Adam optimizer
         ↓
5. Entraînement avec tracking MLflow
         ↓
6. Évaluation sur le jeu de test
         ↓
7. Sauvegarde du modèle et logging dans MLflow
         ↓
8. Visualisation des résultats via MLflow UI
```

## 🔧 Concepts Clés Implémentés

### 1. Deep Learning
- **Réseaux de neurones multicouches** (Dense layers)
- **Fonctions d'activation** (ReLU, Softmax)
- **Régularisation** (Dropout)
- **Optimisation stochastique** (Adam optimizer)
- **Backpropagation** pour la mise à jour des poids

### 2. MLflow Tracking
- **Experiment tracking** : Organisation des runs par expérience
- **Parameter logging** : Enregistrement automatique des hyperparamètres
- **Metric logging** : Suivi des performances par époque
- **Model registry** : Versioning et gestion des modèles
- **Artifact storage** : Sauvegarde des modèles et fichiers

### 3. Bonnes Pratiques ML
- **Séparation train/validation/test** pour évaluation robuste
- **Normalisation des données** pour accélérer la convergence
- **Vectorisation** des opérations pour optimiser les calculs
- **Batch processing** pour gérer efficacement la mémoire

## Améliorations Possibles

### Architecture
- [ ] Ajouter des couches cachées supplémentaires
- [ ] Expérimenter avec des CNN (Convolutional Neural Networks)
- [ ] Tester différentes fonctions d'activation (LeakyReLU, ELU)
- [ ] Implémenter Batch Normalization

### Entraînement
- [ ] Augmenter le nombre d'époques (10-20)
- [ ] Implémenter Early Stopping pour éviter le surapprentissage
- [ ] Ajouter des callbacks (ModelCheckpoint, ReduceLROnPlateau)
- [ ] Expérimenter avec Learning Rate Scheduling

### Optimisation
- [ ] Hyperparameter tuning avec Optuna ou Hyperopt
- [ ] Grid Search ou Random Search sur les hyperparamètres
- [ ] Utiliser des techniques d'augmentation de données

### Déploiement
- [ ] Créer une API REST avec Flask/FastAPI
- [ ] Conteneurisation avec Docker
- [ ] Déploiement sur Cloud (GCP, AWS, Azure)
- [ ] Mise en place d'un pipeline CI/CD

## 📊 Utilisation Avancée de MLflow

### Comparer plusieurs runs

```python
# Lancer plusieurs expérimentations avec différents hyperparamètres
hidden_units_options = [128, 256, 512, 1024]
dropout_rates = [0.1, 0.2, 0.3, 0.5]

for hidden in hidden_units_options:
    for dropout in dropout_rates:
        with mlflow.start_run():
            # Entraîner avec ces paramètres
            mlflow.log_param("hidden_units", hidden)
            mlflow.log_param("dropout_rate", dropout)
            # ... reste du code
```

### Charger un modèle depuis MLflow

```python
import mlflow.keras

# Charger le meilleur modèle
model_uri = "models:/MNIST_DNN_Model/Production"
loaded_model = mlflow.keras.load_model(model_uri)

# Faire des prédictions
predictions = loaded_model.predict(x_test[:10])
```

### Filtrer les runs dans l'API MLflow

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("MNIST_Classification")

# Récupérer les meilleurs runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.test_accuracy > 0.97",
    order_by=["metrics.test_accuracy DESC"]
)
```

##  Débogage et Troubleshooting

### Problème : "No module named 'mlflow'"
```bash
pip install mlflow
```

### Problème : Port 5000 déjà utilisé
```bash
mlflow ui --port 5001
```

### Problème : Modèle non trouvé
Assurez-vous que le fichier `mnist_model.h5` existe dans le répertoire courant.

### Problème : Performances faibles
- Vérifiez la normalisation des données
- Augmentez le nombre d'époques
- Essayez différents learning rates

##  Ressources et Références

- [Documentation TensorFlow/Keras](https://www.tensorflow.org/guide/keras)
- [Documentation MLflow](https://mlflow.org/docs/latest/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Adam Optimizer Paper](https://arxiv.org/abs/1412.6980)

##  Auteur

Projet réalisé dans le cadre d'un TP sur les réseaux de neurones et le Deep Learning.

**Sujet :** De la conception au déploiement de modèles de Deep Learning

##  Licence

Ce projet est à usage éducatif.

---

##  Concepts Pédagogiques Couverts

**Fondamentaux du Deep Learning**
- Descente de gradient stochastique (SGD)
- Backpropagation
- Fonctions d'activation (ReLU, Softmax)
- Régularisation (Dropout)

 **Optimisation**
- Adam optimizer
- Vectorisation des calculs
- Mini-batch processing

 **MLOps et Bonnes Pratiques**
- Experiment tracking avec MLflow
- Versioning des modèles
- Reproductibilité des expérimentations
- Model registry

 **Évaluation de Modèles**
- Séparation train/validation/test
- Métriques de classification
- Monitoring des performances

---

**Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub !** 

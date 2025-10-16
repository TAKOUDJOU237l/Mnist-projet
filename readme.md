# Classification de chiffres manuscrits avec MNIST

## Description du projet

Ce projet implémente un réseau de neurones profond (Deep Neural Network) pour classifier des chiffres manuscrits en utilisant le célèbre dataset MNIST. Le modèle est capable de reconnaître des chiffres de 0 à 9 avec une haute précision.

## Dataset

**MNIST (Modified National Institute of Standards and Technology)**
- 60 000 images d'entraînement
- 10 000 images de test
- Images en niveaux de gris de 28×28 pixels
- 10 classes (chiffres de 0 à 9)

## Architecture du modèle

Le réseau de neurones est composé de :

1. **Couche d'entrée** : 784 neurones (28×28 pixels aplatis)
2. **Couche cachée Dense** : 512 neurones avec activation ReLU
3. **Couche Dropout** : 20% de dropout pour la régularisation
4. **Couche de sortie** : 10 neurones avec activation Softmax (une probabilité par classe)

## Prétraitement des données

- **Normalisation** : Les valeurs des pixels sont normalisées entre 0 et 1 (division par 255)
- **Redimensionnement** : Les images 28×28 sont aplaties en vecteurs de 784 dimensions
- **Conversion** : Les données sont converties en float32 pour l'entraînement

## Configuration de l'entraînement

- **Optimiseur** : Adam (apprentissage adaptatif)
- **Fonction de perte** : Sparse Categorical Crossentropy
- **Métrique** : Accuracy
- **Nombre d'époques** : 5
- **Taille de batch** : 128
- **Validation split** : 10% des données d'entraînement

## Prérequis

```bash
pip install tensorflow numpy
```

## Utilisation

### Entraîner le modèle

```bash
python train_model.py
```

Le script va :
1. Charger automatiquement le dataset MNIST
2. Prétraiter les données
3. Construire et compiler le modèle
4. Entraîner le modèle sur 5 époques
5. Évaluer les performances sur le jeu de test
6. Sauvegarder le modèle entraîné dans `mnist_model.h5`

### Résultats attendus

Le modèle atteint généralement une précision d'environ **97-98%** sur le jeu de test après 5 époques.

## Structure du code

```python
# 1. Chargement des données
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Normalisation et redimensionnement
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape(60000, 784)

# 3. Construction du modèle
model = keras.Sequential([...])

# 4. Compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# 5. Entraînement
history = model.fit(x_train, y_train, epochs=5, batch_size=128)

# 6. Évaluation
test_loss, test_acc = model.evaluate(x_test, y_test)

# 7. Sauvegarde
model.save('mnist_model.h5')
```

## Concepts clés utilisés

### Techniques de Deep Learning
- **Fully Connected Network** : Architecture simple mais efficace
- **ReLU Activation** : Fonction d'activation non-linéaire pour les couches cachées
- **Softmax** : Transformation des sorties en probabilités
- **Dropout** : Régularisation pour éviter le surapprentissage

### Optimisations
- **Vectorisation** : Opérations matricielles optimisées
- **Mini-batch processing** : Traitement par lots pour accélérer l'entraînement
- **Adam Optimizer** : Convergence rapide et robuste

## Améliorations possibles

1. Augmenter le nombre d'époques pour améliorer la précision
2. Ajouter des couches cachées supplémentaires
3. Expérimenter avec différents taux de dropout
4. Utiliser des réseaux convolutifs (CNN) pour de meilleures performances
5. Implémenter des techniques d'augmentation de données
6. Ajouter un Early Stopping pour éviter le surapprentissage

## Fichiers générés

- `mnist_model.h5` : Modèle entraîné sauvegardé au format HDF5

## Chargement du modèle sauvegardé

```python
from tensorflow import keras

# Charger le modèle
model = keras.models.load_model('mnist_model.h5')

# Faire des prédictions
predictions = model.predict(x_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
```

## Auteur

Projet réalisé dans le cadre d'un TP sur les réseaux de neurones et le Deep Learning.

## Licence

Ce projet est à usage éducatif.
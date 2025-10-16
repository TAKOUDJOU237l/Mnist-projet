import tensorflow as tf
from tensorflow.keras import keras
import numpy as np

# chargement du jeu de données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#normalisation des données
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# redimensionnement des images  pour  les reseaux fullly connected
x_train = x_train.reshape(60000 ,784)
x_test = x_test.reshape(10000 ,784)

# construction du modèle
model = keras.Sequential([
    
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# compilation du modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# entraînement du modèle
history = model.fit(
    x_train, 
    y_train, 
    epochs=5,
    batch_size=128,
    validation_split=0.1)

# évaluation du modèle
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"précision sur les donnes: {test_acc :.4f}")

# sauvegarde du modèle
model.save('mnist_model.h5')
print("Modèle sauvegardé sous le nom 'mnist_model.h5'")

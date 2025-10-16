import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.keras
from mlflow.models import infer_signature

# Configuration de l'expérimentation MLflow
mlflow.set_experiment("MNIST_Classification")

# Démarrer un run MLflow
with mlflow.start_run():
    
    # Définir les hyperparamètres
    epochs = 5
    batch_size = 128
    validation_split = 0.1
    hidden_units = 512
    dropout_rate = 0.2
    optimizer = 'adam'
    
    # Logger les hyperparamètres
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("validation_split", validation_split)
    mlflow.log_param("hidden_units", hidden_units)
    mlflow.log_param("dropout_rate", dropout_rate)
    mlflow.log_param("optimizer", optimizer)
    mlflow.log_param("activation_hidden", "relu")
    mlflow.log_param("activation_output", "softmax")
    
    # Chargement du jeu de données MNIST
    print("Chargement des données MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalisation des données
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Redimensionnement des images pour les réseaux fully connected
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    
    # Logger les informations sur le dataset
    mlflow.log_param("train_samples", x_train.shape[0])
    mlflow.log_param("test_samples", x_test.shape[0])
    mlflow.log_param("input_shape", x_train.shape[1])
    mlflow.log_param("num_classes", 10)
    
    # Construction du modèle
    print("Construction du modèle...")
    model = keras.Sequential([
        keras.layers.Dense(hidden_units, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Afficher le résumé du modèle
    model.summary()
    
    # Logger le nombre de paramètres
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    mlflow.log_param("trainable_parameters", int(trainable_params))
    
    # Compilation du modèle
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entraînement du modèle
    print("Entraînement du modèle...")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    
    # Logger les métriques d'entraînement pour chaque époque
    for epoch in range(epochs):
        mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
    
    # Évaluation du modèle sur le jeu de test
    print("\nÉvaluation du modèle sur le jeu de test...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    
    # Logger les métriques finales
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)
    
    print(f"\nPrécision sur les données de test : {test_acc:.4f}")
    print(f"Perte sur les données de test : {test_loss:.4f}")
    
    # Créer la signature du modèle pour MLflow
    signature = infer_signature(x_test[:5], model.predict(x_test[:5]))
    
    # Logger le modèle avec MLflow
    print("\nSauvegarde du modèle avec MLflow...")
    mlflow.keras.log_model(
        model,
        "model",
        signature=signature,
        registered_model_name="MNIST_DNN_Model"
    )
    
    # Sauvegarder également au format classique
    model.save('mnist_model.h5')
    mlflow.log_artifact('mnist_model.h5')
    
    # Logger des tags pour faciliter la recherche
    mlflow.set_tag("model_type", "Dense Neural Network")
    mlflow.set_tag("dataset", "MNIST")
    mlflow.set_tag("framework", "TensorFlow/Keras")
    
    print("\nModèle sauvegardé avec MLflow ")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("\nPour visualiser les résultats, lancez: mlflow ui")
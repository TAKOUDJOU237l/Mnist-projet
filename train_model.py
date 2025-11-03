import tensorflow as tf
from tensorflow import keras
import numpy as np
import mlflow
import mlflow.keras
from mlflow.models import infer_signature
import matplotlib.pyplot as plt

# Configuration de l'expérimentation MLflow
mlflow.set_experiment("MNIST_Classification")

# Définir les hyperparamètres communs
epochs = 5
batch_size = 128
hidden_units = 512
dropout_rate = 0.2

# Définir les optimiseurs à tester
optimizers_to_test = [
    {'name': 'adam', 'optimizer': 'adam'},
    {'name': 'sgd', 'optimizer': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)},
    {'name': 'rmsprop', 'optimizer': 'rmsprop'}
]

# Chargement du jeu de données MNIST (une seule fois)
print("Chargement du jeu de données MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Utilisez 90% pour l'entraînement et 10% pour la validation (dev)
x_val = x_train[54000:]
y_val = y_train[54000:]
x_train = x_train[:54000]
y_train = y_train[:54000]

# Normalisation et remodelage comme dans le TP 1
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape(54000, 784)
x_val = x_val.reshape(6000, 784)
x_test = x_test.reshape(10000, 784)

print(f"Données préparées: Train={x_train.shape}, Val={x_val.shape}, Test={x_test.shape}\n")

# Boucle sur les différents optimiseurs
for optimizer_config in optimizers_to_test:
    
    optimizer_name = optimizer_config['name']
    optimizer = optimizer_config['optimizer']
    
    print(f"\n{'='*70}")
    print(f"ENTRAÎNEMENT AVEC L'OPTIMISEUR: {optimizer_name.upper()}")
    print(f"{'='*70}\n")
    
    # Démarrer un run MLflow pour chaque optimiseur
    with mlflow.start_run(run_name=f"MNIST_{optimizer_name}"):
        
        # Logger les hyperparamètres
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("hidden_units", hidden_units)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("optimizer", optimizer_name)
        mlflow.log_param("activation_hidden", "relu")
        mlflow.log_param("activation_output", "softmax")
        mlflow.log_param("l2_regularization", 0.001)
        
        # Logger les paramètres spécifiques pour SGD
        if optimizer_name == 'sgd':
            mlflow.log_param("learning_rate", 0.01)
            mlflow.log_param("momentum", 0.9)
        
        # Logger les informations sur le dataset
        mlflow.log_param("train_samples", x_train.shape[0])
        mlflow.log_param("val_samples", x_val.shape[0])
        mlflow.log_param("test_samples", x_test.shape[0])
        mlflow.log_param("input_shape", x_train.shape[1])
        mlflow.log_param("num_classes", 10)
    
        mlflow.log_param("num_classes", 10)
        
        # Construction du modèle
        print("Construction du modèle...")
        model = keras.Sequential([
            keras.layers.Dense(hidden_units, activation='relu', input_shape=(784,), 
                             
                              kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
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
        print(f"Entraînement du modèle avec {optimizer_name}...")
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Logger les métriques d'entraînement pour chaque époque
        for epoch in range(epochs):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Afficher les résultats de validation
        print("\n=== Résultats de validation par époque ===")
        for epoch in range(epochs):
            print(f"Époque {epoch+1}/{epochs}:")
            print(f"  Train Loss: {history.history['loss'][epoch]:.4f} - Train Acc: {history.history['accuracy'][epoch]:.4f}")
            print(f"  Val Loss: {history.history['val_loss'][epoch]:.4f} - Val Acc: {history.history['val_accuracy'][epoch]:.4f}")
        
        # Tracer les courbes d'apprentissage
        print("\nGénération des courbes d'apprentissage...")
        
        # Créer une figure avec deux sous-graphiques
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Graphique 1: Précision (Accuracy)
        ax1.plot(history.history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title(f'Précision du modèle - {optimizer_name}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Époque', fontsize=12)
        ax1.set_ylabel('Précision', fontsize=12)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Perte (Loss)
        ax2.plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title(f'Perte du modèle - {optimizer_name}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Époque', fontsize=12)
        ax2.set_ylabel('Perte', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder la figure
        learning_curves_path = f'learning_curves_{optimizer_name}.png'
        plt.savefig(learning_curves_path, dpi=300, bbox_inches='tight')
        print(f"Courbes d'apprentissage sauvegardées : {learning_curves_path}")
        
        # Logger la figure dans MLflow
        mlflow.log_artifact(learning_curves_path)
        
        # Afficher la figure
        plt.show()
        plt.close()
        
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
            registered_model_name=f"MNIST_DNN_{optimizer_name}"
        )
        
        # Sauvegarder également au format classique
        model_path = f'mnist_model_{optimizer_name}.h5'
        model.save(model_path)
        mlflow.log_artifact(model_path)
        
        # Logger des tags pour faciliter la recherche
        mlflow.set_tag("model_type", "Dense Neural Network")
        mlflow.set_tag("dataset", "MNIST")
        mlflow.set_tag("framework", "TensorFlow/Keras")
        mlflow.set_tag("optimizer_type", optimizer_name)
        
        print(f"\nModèle sauvegardé avec MLflow")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

print("\n" + "="*70)
print("TOUTES LES EXPÉRIENCES SONT TERMINÉES!")
print("="*70)
print("\nPour visualiser et comparer les résultats, lancez: mlflow ui")
print("Vous pourrez comparer les performances des différents optimiseurs dans l'interface MLflow.")
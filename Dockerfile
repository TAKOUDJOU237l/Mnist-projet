# 1. Image de base officielle Python
FROM python:3.11-slim

# 2. Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier les fichiers nécessaires
COPY requirements.txt .
COPY app.py .
COPY mnist_model.h5 .


# 4. Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Exposer le port utilisé par Flask
EXPOSE 5000

# 6. Définir la commande de démarrage
CMD ["python", "app.py"]

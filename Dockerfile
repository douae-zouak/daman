# Utilisation d'une image Python légère
FROM python:3.11-slim

# Éviter la génération de fichiers .pyc et activer le buffering des logs
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Installation des dépendances système nécessaires pour PaddleOCR et OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Définition du dossier de travail
WORKDIR /app

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances Python
# Note: On utilise --no-cache-dir pour réduire la taille de l'image
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code source
COPY . .

# Création des dossiers de données pour éviter des problèmes de permissions
RUN mkdir -p data/uploads data/documents data/faq chroma_db

# Exposition du port utilisé par FastAPI
EXPOSE 8000

# Commande de lancement
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

"""
Fournit le modèle d'embedding multilingue (français + arabe).
Utilise sentence-transformers en local → aucun coût API.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Retourne le modèle d'embedding multilingue.

    Modèle recommandé : intfloat/multilingual-e5-base
    ─ Supporte 100+ langues dont le français et l'arabe
    ─ Bonne performance sur les textes administratifs courts
    ─ Tourne en local sur CPU (pas besoin de GPU)

    Alternative plus légère : paraphrase-multilingual-MiniLM-L12-v2
    """
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
        # normalise chaque vecteur pour avoir une longueur de 1, ce qui permet d'utiliser la similarité cosine (mesure d'angle). Sinon il faudrait utiliser la distance euclidienne.
    )
    return model


    # chunk 1 = "Pour obtenir un passeport..."
        #             ↓  E5 (multilingual-e5-base)
        #             ↓  réseau de neurones (transformer, 12 couches)
        #   [0.12, -0.45, 0.78, 0.03, 0.61, ..., 0.91]
        #    ←──────────── 768 nombres réels ────────────→

# Oui, c'est bien E5 qui vectorise, via HuggingFaceEmbeddings. Il prend du texte en entrée et sort 768 nombres. C'est un réseau de neurones de 278 millions de paramètres qui a "appris" le sens des mots dans 100+ langues.

# Chaque chunk devient donc un point dans un espace vectoriel à 768 dimensions.
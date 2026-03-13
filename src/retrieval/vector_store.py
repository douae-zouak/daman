"""
Interface avec ChromaDB (base vectorielle locale, persistante, gratuite).
Deux collections séparées : une pour les guides TXT, une pour la FAQ.
"""

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config.settings import CHROMA_DIR


COLLECTION_GUIDES = "guides_administratifs"
COLLECTION_FAQ    = "faq"


def build_vector_store(documents: list[Document], embedding_model, collection_name: str) -> Chroma:

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )
    print(f"[ChromaDB] Collection '{collection_name}' → {len(documents)} documents indexés")
    return vectorstore


# Appelle E5 sur le page_content → produit un vecteur de 768 floats
# Stocke le vecteur + les metadata sur disque dans CHROMA_DIR

def load_vector_store(embedding_model, collection_name: str) -> Chroma:
    """
    Charge une collection ChromaDB existante depuis le disque.
    À utiliser à chaque démarrage de l'application (après l'ingestion initiale).
    """
    vectorstore = Chroma(
        collection_name=collection_name,
    embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore
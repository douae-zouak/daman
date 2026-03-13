"""
src/pipeline.py
────────────────
Orchestrateur central du pipeline RAG.
Deux modes :
  - build()  : ingestion initiale (à lancer une seule fois)
  - query()  : répondre à une question (usage normal)
"""

from src.ingestion.txt_loader   import load_txt_documents
from src.ingestion.faq_loader   import load_faq_documents
from src.ingestion.embedder     import get_embedding_model
from src.retrieval.vector_store import build_vector_store, load_vector_store, COLLECTION_GUIDES, COLLECTION_FAQ
from src.retrieval.retriever    import HybridRetriever
from src.generation.generator   import RAGGenerator


class RAGPipeline:

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.guide_store     = None
        self.faq_store       = None
        self.retriever       = None
        self.generator       = RAGGenerator()

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 1 — INGESTION (une seule fois)
    # ─────────────────────────────────────────────────────────────────────────

    def build(self):
        """
        Charge les données brutes, génère les embeddings et persiste ChromaDB.
        À lancer UNE SEULE FOIS (ou après mise à jour des données).
        """
        print("=" * 50)
        print("INGESTION — Construction de la base vectorielle")
        print("=" * 50)

        # 1. Chargement des documents
        print("Chargement des TXT...")
        guide_docs = load_txt_documents()
        print("Chargement FAQ...")
        faq_docs   = load_faq_documents()

        # 2. Indexation dans ChromaDB

        print("Construction Chroma guides...")
        self.guide_store = build_vector_store(guide_docs, self.embedding_model, COLLECTION_GUIDES)
        print("Construction Chroma FAQ...")
        self.faq_store   = build_vector_store(faq_docs,   self.embedding_model, COLLECTION_FAQ)

        # 3. Initialisation du retriever
        self.retriever = HybridRetriever(self.guide_store, self.faq_store)

        print("\n✅ Ingestion terminée. Base vectorielle prête.")

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 2 — CHARGEMENT (à chaque démarrage)
    # ─────────────────────────────────────────────────────────────────────────

    def load(self):
        """
        Charge la base vectorielle existante depuis le disque.
        À appeler à chaque démarrage de l'application (après un premier build()).
        """
        self.guide_store = load_vector_store(self.embedding_model, COLLECTION_GUIDES)
        self.faq_store   = load_vector_store(self.embedding_model, COLLECTION_FAQ)
        self.retriever   = HybridRetriever(self.guide_store, self.faq_store)
        print("✅ Base vectorielle chargée depuis le disque.")

    # ─────────────────────────────────────────────────────────────────────────
    # ÉTAPE 3 — REQUÊTE (usage normal)
    # ─────────────────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Pipeline complet pour une question :
          1. Retrieval hybride (FAQ + guides)
          2. Génération de la réponse (directe ou via LLM)
        """
        if self.retriever is None:
            raise RuntimeError("Pipeline non initialisé. Appelez build() ou load() d'abord.")

        print(f"\n[Query] {question}")

        # Retrieval
        retrieval_result = self.retriever.retrieve(question)

        # Génération
        result = self.generator.generate(question, retrieval_result)

        return result
"""
Logique de recherche HYBRIDE :
  1. Recherche sémantique dans la FAQ  → réponse directe si score suffisant
  2. Recherche sémantique dans les guides TXT → contexte pour le LLM
  3. Fallback par mots-clés si la recherche sémantique est insuffisante
"""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config.settings import TOP_K_DOCS, TOP_K_FAQ, FAQ_SCORE_THRESHOLD


class HybridRetriever:
    def __init__(self, guide_store: Chroma, faq_store: Chroma):
        self.guide_store = guide_store
        self.faq_store   = faq_store

    def retrieve(self, query: str) -> dict:
        """
        Retourne :
          - faq_match  : Document FAQ si trouvé avec score > seuil, sinon None
          - guide_docs : liste des chunks TXT pertinents
        """
        # 1. Recherche dans la FAQ 
        faq_results = self.faq_store.similarity_search_with_score(query, k=TOP_K_FAQ)

        faq_match = None
        for doc, score in faq_results:
            # ChromaDB retourne une distance L2 (plus petit = meilleur)
            # On la convertit en similarité cosine normalisée (0→1)
            similarity = 1 - score
            if similarity >= FAQ_SCORE_THRESHOLD:
                faq_match = doc
                print(f"[Retriever] FAQ match → '{doc.metadata['question_fr']}' (score={similarity:.2f})")
                break

        # ── 2. Recherche dans les guides TXT ─────────────────────────────────
        guide_docs = self.guide_store.similarity_search(query, k=TOP_K_DOCS)
        print(f"[Retriever] {len(guide_docs)} chunks guides récupérés")

        return {
            "faq_match":  faq_match,
            "guide_docs": guide_docs,
        }
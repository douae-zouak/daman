"""
src/generation/generator.py
────────────────────────────
Construit le prompt RAG et appelle le LLM (OpenAI ou Mistral).
Si un match FAQ exact est trouvé → réponse directe (sans LLM).
Sinon → le LLM synthétise la réponse depuis les chunks guides.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from config.settings import LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_TEMPERATURE


# ─── Chargement du LLM selon le fournisseur ──────────────────────────────────

def get_llm():
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url="http://localhost:11434",
        )
    elif LLM_PROVIDER == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=LLM_API_KEY,
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=LLM_API_KEY,
        )
    else:
        raise ValueError(f"LLM_PROVIDER inconnu : {LLM_PROVIDER}")


# ─── Prompt RAG ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant administratif spécialisé dans les démarches administratives au Maroc.
Tu aides les citoyens marocains à comprendre les procédures, les documents requis et les démarches à suivre.

Règles strictes :
- Réponds UNIQUEMENT à partir du contexte fourni ci-dessous.
- Si l'information n'est pas dans le contexte, dis-le clairement : "Je ne dispose pas de cette information."
- Sois précis, clair et bienveillant. Utilise des listes numérotées si nécessaire.
- Si la question est en arabe, réponds en arabe. Sinon, réponds en français.
- Ne jamais inventer de montants, de délais ou de listes de documents.

Contexte administratif :
─────────────────────────
{context}
─────────────────────────
"""

HUMAN_PROMPT = "Question du citoyen : {question}"

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ─── Générateur principal ─────────────────────────────────────────────────────

class RAGGenerator:
    def __init__(self):
        self.llm = get_llm()

    def generate(self, query: str, retrieval_result: dict) -> dict:
        """
        Reçoit les résultats du retriever et produit une réponse finale.
        Retourne : { "answer": str, "source": "faq" | "llm", "metadata": dict }
        """
        faq_match  = retrieval_result.get("faq_match")
        guide_docs = retrieval_result.get("guide_docs", [])

        # ── CAS 1 : Match FAQ direct → réponse immédiate, pas de LLM ─────────
        if faq_match:
            lang = self._detect_language(query)
            answer = (
                faq_match.metadata["reponse_ar"]
                if lang == "ar" and faq_match.metadata.get("reponse_ar")
                else faq_match.metadata["reponse_fr"]
            )
            return {
                "answer":   answer,
                "source":   "faq",
                "metadata": {
                    "faq_id":    faq_match.metadata["id"],
                    "categorie": faq_match.metadata["categorie"],
                },
            }

        # ── CAS 2 : Pas de FAQ → LLM + contexte guides ───────────────────────
        if not guide_docs:
            return {
                "answer": "Je ne dispose pas d'informations sur ce sujet dans ma base de connaissances.",
                "source": "no_context",
                "metadata": {},
            }

        context = self._format_context(guide_docs)
        chain   = rag_prompt | self.llm

        response = chain.invoke({"context": context, "question": query})

        return {
            "answer":   response.content,
            "source":   "llm",
            "metadata": {
                "sources": list({doc.metadata["source"] for doc in guide_docs}),
                "themes":  list({doc.metadata.get("theme", "") for doc in guide_docs}),
            },
        }

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "inconnu")
            parts.append(f"[Source {i} — {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _detect_language(text: str) -> str:
        """Détection simple : présence de caractères arabes."""
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        return "ar" if arabic_chars > len(text) * 0.3 else "fr"
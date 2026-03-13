"""
main.py — Point d'entrée du projet
────────────────────────────────────
Usage :
  python main.py --build          # Ingestion initiale (une seule fois)
  python main.py                  # Mode interactif (questions/réponses)
  python main.py --query "..."    # Répondre à une question unique
"""

import argparse
from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Assistant Administratif Maroc — RAG Pipeline")
    parser.add_argument("--build", action="store_true", help="Lancer l'ingestion initiale")
    parser.add_argument("--query", type=str, default=None, help="Poser une question directement")
    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.build:
        pipeline.build()
        return

    pipeline.load()

    if args.query:
        result = pipeline.query(args.query)
        print_result(result)
        return

    # ── Mode interactif ───────────────────────────────────────────────────────
    print("\n🇲🇦 Assistant Administratif Maroc")
    print("Tapez votre question (ou 'quitter' pour arrêter)\n")

    while True:
        question = input("Vous : ").strip()
        if not question:
            continue
        if question.lower() in ("quitter", "exit", "quit"):
            break

        result = pipeline.query(question)
        print_result(result)


def print_result(result: dict):
    print("\n" + "─" * 50)
    print(f"Source   : {result['source'].upper()}")
    print(f"Réponse  :\n{result['answer']}")
    if result.get("metadata"):
        print(f"Metadata : {result['metadata']}")
    print("─" * 50 + "\n")


if __name__ == "__main__":
    main()
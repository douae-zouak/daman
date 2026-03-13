import json
from langchain_core.documents import Document
from config.settings import FAQ_JSON_PATH


def load_faq_documents() -> list[Document]:

    with open(FAQ_JSON_PATH, "r", encoding="utf-8") as f:
        faq_list = json.load(f)

    documents: list[Document] = []

    for item in faq_list:
        content = f"Question : {item['question_fr']}\nRéponse : {item['answer_fr']}"

        metadata = {
            "id":          item["id"],
            "type":        "faq",
            "categorie":   item.get("category", ""),
            "source":      item.get("source", ""),
            "mots_cles":   ", ".join(item.get("keywords", [])),
            "question_fr": item["question_fr"],
            "reponse_fr":  item["answer_fr"],
            "question_ar": item.get("question_ar", ""),
            "reponse_ar":  item.get("answer_ar", ""),
        }

        documents.append(Document(page_content=content, metadata=metadata))

    print(f"[FAQ] {len(documents)} entrées chargées")
    return documents
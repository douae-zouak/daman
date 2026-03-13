import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import TXT_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_txt_documents() -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    all_chunks: list[Document] = []

    for filename in os.listdir(TXT_DIR):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(TXT_DIR, filename)
        theme    = os.path.splitext(filename)[0]

        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        chunks = splitter.create_documents(
            texts=[raw_text],
            metadatas=[{"source": filename, "theme": theme, "type": "guide"}],
        )
        all_chunks.extend(chunks)
        print(f"[TXT] {filename} → {len(chunks)} chunks")

    return all_chunks
# ingest.py
import os
import json
import datetime
from pathlib import Path
from hashlib import md5
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BOOKS_FOLDER = r"D:\GenAI Learning\GenAiCodeRelated\RAG\books"
CHROMA_PATH = Path("./chroma_db_offline")
CHROMA_PATH.mkdir(exist_ok=True)
CACHE_FILE = CHROMA_PATH / "content_hashes.json"  # Dedup by content

# Supported file types
FILE_LOADERS = {
    ".md": DirectoryLoader,
    ".txt": DirectoryLoader,
    ".pdf": lambda path: [PyPDFLoader(path).load()],
}

# -------------------------------------------------
# 1. Load embedding model (100% offline)
# -------------------------------------------------
print("Loading embedding model...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},  # change to "cuda" if GPU
    encode_kwargs={"normalize_embeddings": True},
)

# -------------------------------------------------
# 2. Connect to existing Chroma DB
# -------------------------------------------------
vectorstore = Chroma(
    collection_name="books_rag_offline",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH),
)

# -------------------------------------------------
# 3. Load deduplication cache (content-based)
# -------------------------------------------------
def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.load(open(CACHE_FILE))
    return {}

def save_cache(cache: dict):
    json.dump(cache, open(CACHE_FILE, "w"), indent=2)

cache = load_cache()  # { "hash": { "filename": ..., "added_at": ... } }

# -------------------------------------------------
# 4. Load all files (support .md, .txt, .pdf)
# -------------------------------------------------
def load_file(file_path: Path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    elif ext in [".md", ".txt"]:
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()
    return []

all_docs = []
for ext in FILE_LOADERS.keys():
    pattern = f"**/*{ext}"
    if ext == ".pdf":
        for pdf_path in Path(BOOKS_FOLDER).rglob("*.pdf"):
            all_docs.extend(load_file(pdf_path))
    else:
        loader = DirectoryLoader(BOOKS_FOLDER, glob=pattern)
        all_docs.extend(loader.load())

print(f"Found {len(all_docs)} documents")

# -------------------------------------------------
# 5. Deduplicate & detect changes
# -------------------------------------------------
def file_content_hash(doc) -> str:
    source = doc.metadata["source"]
    path = Path(source)
    return md5(path.read_bytes()).hexdigest() if path.exists() else ""

new_docs = []
seen_hashes = set()

for doc in all_docs:
    content_hash = file_content_hash(doc)
    if not content_hash:
        continue

    if content_hash not in cache:
        print(f"New content: {Path(doc.metadata['source']).name}")
        new_docs.append(doc)
        cache[content_hash] = {
            "filename": Path(doc.metadata["source"]).name,
            "added_at": datetime.datetime.now().isoformat()
        }
    else:
        print(f"Duplicate (skip): {Path(doc.metadata['source']).name}")
    seen_hashes.add(content_hash)

# -------------------------------------------------
# 6. Chunk only new docs
# -------------------------------------------------
if not new_docs:
    print("No new content. Database is up to date.")
else:
    print(f"Chunking {len(new_docs)} new documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(new_docs)
    print(f"Adding {len(chunks)} new chunks...")

    # -------------------------------------------------
    # 7. ADD to existing DB (no delete!)
    # -------------------------------------------------
    vectorstore.add_documents(chunks)
    print(f"Success! Total chunks in DB: {vectorstore._collection.count()}")

    save_cache(cache)

print(f"Vector DB: {CHROMA_PATH}")
print(f"Cache: {CACHE_FILE}")
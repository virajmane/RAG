# ingest.py
import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- 1. OFFLINE EMBEDDINGS ----------
# Use a small, fully-open model that runs 100% locally
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# ---------- 2. VECTOR STORE (local) ----------
from langchain_chroma import Chroma

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BOOKS_FOLDER = r"D:\GenAI Learning\GenAiCodeRelated\RAG\langchain-rag-tutorial\data\books"
CHROMA_PATH = Path("./chroma_db_offline")
CHROMA_PATH.mkdir(exist_ok=True)

# 1. Load documents
loader = DirectoryLoader(BOOKS_FOLDER, glob="**/*.md")
docs = loader.load()
print(f"Loaded {len(docs)} documents")

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# 3. OFFLINE embeddings (BGE-small-en-v1.5 – ~135 MB, fast & accurate)
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",   # download once → cached in ~/.cache/huggingface
    model_kwargs={"device": "cpu"},        # change to "cuda" if you have a GPU
    encode_kwargs={"normalize_embeddings": True},
)

# 4. Build / persist Chroma DB locally
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="books_rag_offline",
    persist_directory=str(CHROMA_PATH),
)
print(f"Vector DB saved to {CHROMA_PATH}")
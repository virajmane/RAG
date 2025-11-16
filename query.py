# query.py
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------
# 1. Load .env
# -------------------------------------------------
load_dotenv()  # reads .env → sets os.environ

# -------------------------------------------------
# 2. CONFIG
# -------------------------------------------------
CHROMA_PATH = Path("./chroma_db_offline")
QUESTION = "What is the main character's name in Alice in Wonderland?"

# -------------------------------------------------
# 3. Offline embeddings
# -------------------------------------------------
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# -------------------------------------------------
# 4. Load vector DB
# -------------------------------------------------
vectorstore = Chroma(
    collection_name="books_rag_offline",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# -------------------------------------------------
# 5. Retrieve (offline)
# -------------------------------------------------
print("\n=== RETRIEVED CHUNKS (offline) ===")
retrieved = retriever.invoke(QUESTION)
for i, doc in enumerate(retrieved, 1):
    src = Path(doc.metadata.get("source", "unknown")).name
    print(f"\n--- Chunk {i} [{src}] ---\n{doc.page_content}\n")

context_str = "\n\n".join(
    f"[Chunk {i+1}] {d.page_content}" for i, d in enumerate(retrieved)
)

# -------------------------------------------------
# 6. Prompt
# -------------------------------------------------
prompt = PromptTemplate.from_template(
    """You are a strict assistant. Answer using **only** the context below.
If the answer is not in the context, reply: **"I don't know."**

Context:
{context}

Question: {question}
Answer:"""
)

filled_prompt = prompt.format(context=context_str, question=QUESTION)
print("\n=== PROMPT SENT TO LLM ===")
print(filled_prompt)
print("\n" + "=" * 80)

# -------------------------------------------------
# 7. LLM – **CRITICAL FIX**
# -------------------------------------------------
api_key = os.getenv("OPENROUTER_API_KEY")
api_base = os.getenv("OPENROUTER_API_BASE")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file!")

# Force LangChain to treat the key as a plain string (not a callable)
llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",   # or "openai/gpt-4o-mini"
    temperature=0.0,
    timeout=30,
    max_retries=2,
    api_key=api_key,
    base_url=api_base,
)

print("\nCalling OpenRouter...")
raw_response = llm.invoke(filled_prompt)
print("\n=== RAW LLM RESPONSE ===")
print(raw_response.content)

# -------------------------------------------------
# 8. Final answer
# -------------------------------------------------
parser = StrOutputParser()
answer = parser.parse(raw_response)
print("\n=== FINAL ANSWER ===")
print(answer)
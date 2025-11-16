# query.py  (audit-enabled version)
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
CHROMA_PATH = Path("./chroma_db_offline")
QUESTION = "What is the main character's name in Alice in Wonderland?"

# ---------- 1. OFFLINE EMBEDDINGS ----------
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ---------- 2. LOAD LOCAL VECTOR DB ----------
vectorstore = Chroma(
    collection_name="books_rag_offline",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ---------- 3. RETRIEVE (OFFLINE) ----------
print("\n=== RETRIEVED CHUNKS (offline) ===")
retrieved = retriever.invoke(QUESTION)
for i, doc in enumerate(retrieved, 1):
    print(f"\n--- Chunk {i} ---\n{doc.page_content}\n{'-'*60}")

# Build context string (exactly what will be sent to LLM)
context_str = "\n\n".join([d.page_content for d in retrieved])

# ---------- 4. BUILD PROMPT (you will see the *exact* prompt) ----------
prompt = PromptTemplate.from_template(
    """You are a strict assistant. 
You MUST answer **only** using the context below. 
If the answer is not in the context, reply **"I don't know."**

Context:
{context}

Question: {question}
Answer:"""
)

filled_prompt = prompt.format(context=context_str, question=QUESTION)
print("\n=== EXACT PROMPT SENT TO LLM ===")
print(filled_prompt)
print("\n" + "="*80)

# ---------- 5. CALL OPENROUTER (only online part) ----------
os.environ["OPENAI_API_KEY"] = "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"          # ← YOUR OPENROUTER KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.0,          # deterministic = easier to audit
)

# Get **raw** LLM output (no parser yet)
# print("\nCalling LLM (OpenRouter)...")
# raw_response = llm.invoke(prompt.format(context=context_str, question=QUESTION))
# print("\n=== RAW LLM RESPONSE (content only) ===")
# print(raw_response.content)      # <-- this is what the model *actually* said

# # ---------- 6. OPTIONAL: Clean answer ----------
# parser = StrOutputParser()
# answer = parser.parse(raw_response)
# print("\n=== FINAL ANSWER ===")
# print(answer)

# HALLUCINATION TEST
print("\n" + "="*60)
print("HALLUCINATION TEST: Asking something NOT in the book")
print("="*60)

fake_question = "What is your model name?"

fake_chunks = retriever.invoke(fake_question)
fake_context = "\n\n".join([d.page_content for d in fake_chunks])

test_prompt = prompt.format(context=fake_context, question=fake_question)
print("\nPrompt sent (check: no mention of ice cream):")
print(test_prompt[:500] + "...")

response = llm.invoke(test_prompt)
print("\nLLM Response:", response.content.strip())
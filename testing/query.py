import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set up OpenRouter API (replace with your actual key)
os.environ["OPENAI_API_KEY"] = "your_openrouter_api_key_here"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Step 1: Load existing vector store
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Chroma(collection_name="books_rag", embedding_function=embeddings, persist_directory="./chroma_db")

# Step 2: Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 chunks

# Step 3: Retrieve and inspect chunks
question = "What is the main character's name in Alice in Wonderland?"  # Change this to your query
retrieved_docs = retriever.invoke(question)

print("Retrieved chunks from vector DB:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nChunk {i}:\n{doc.page_content}\n{'-' * 80}")

# Step 4: Set up LLM and RAG chain (optional: comment out if you just want to check retrieval)
llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0.7)

prompt_template = """
You are an assistant answering questions based on the provided context.
Context: {context}
Question: {question}
Answer the question concisely using only the context provided.
"""
prompt = PromptTemplate.from_template(prompt_template)

rag_chain = (
    {"context": RunnablePassthrough() | (lambda q: "\n\n".join([doc.page_content for doc in retriever.invoke(q)])),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example LLM invocation (uncomment to run)
# answer = rag_chain.invoke(question)
# print(f"\nQuestion: {question}")
# print(f"Answer: {answer}")
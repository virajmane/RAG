import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set up OpenRouter API (replace with your actual key)
os.environ["OPENAI_API_KEY"] = "your_openrouter_api_key_here"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Path to the books folder (adjust if needed)
books_folder = "D:\\GenAI Learning\\GenAiCodeRelated\\RAG\\langchain-rag-tutorial\\data\\books"

# Step 1: Load all documents from the books folder (assuming .md files, but can load others)
loader = DirectoryLoader(books_folder, glob="**/*.md")  # Change glob if other file types
docs = loader.load()

# Step 2: Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Step 3: Set up embeddings (using OpenAI-compatible via OpenRouter)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Ensure model is available on OpenRouter

# Step 4: Create vector store (using Chroma as a simple local vector DB)
vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="books_rag")

# Step 5: Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 chunks

# Step 6: Set up LLM (using a model available on OpenRouter, e.g., gpt-4o-mini)
llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0.7)

# Step 7: Define prompt template
prompt_template = """
You are an assistant answering questions based on the provided context.
Context: {context}
Question: {question}
Answer the question concisely using only the context provided.
"""
prompt = PromptTemplate.from_template(prompt_template)

# Step 8: Create the RAG chain
rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join([doc.page_content for doc in docs])),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example usage
if __name__ == "__main__":
    question = "What is the main character's name in Alice in Wonderland?"
    answer = rag_chain.invoke(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
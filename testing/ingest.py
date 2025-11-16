import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

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
vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="books_rag", persist_directory="./chroma_db")
print("Ingestion complete. Vector DB stored in ./chroma_db")
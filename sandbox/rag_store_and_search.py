from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


# 1. Read the text file
with open("data/notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Split into chunks (same as before)
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=150,
    chunk_overlap=30,
    length_function=len,
)

chunks = text_splitter.split_text(text)

print(f"Loaded document with {len(chunks)} chunks.")

# 3. Create an embeddings object using Ollama
#    This will call the local Ollama server to get embeddings.
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Create (or connect to) a Chroma vector store in a local folder "chroma_db"
vector_store = Chroma(
    collection_name="notes_collection",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

# 5. Add chunks to Chroma with simple IDs like "chunk-0", "chunk-1", ...
ids = [f"chunk-{i}" for i in range(len(chunks))]

# Optional: clear existing docs while experimenting
# vector_store.delete(ids=ids)

vector_store.add_texts(texts=chunks, ids=ids)

print("Chunks stored in Chroma.")

# 6. Ask a test question and search for similar chunks
query = "What is RAG?"
print(f"\nQuery: {query}")

results = vector_store.similarity_search(query, k=2)

print("\nTop retrieved chunks:")
for i, doc in enumerate(results, start=1):
    print(f"[Result {i}]")
    print(doc.page_content)
    print("-" * 40)
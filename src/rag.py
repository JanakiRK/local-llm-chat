import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# 1. Read and chunk all documents in ../data
data_dir = Path("../data")
file_extensions = [".txt", ".md"]  # we’ll add PDF later

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=400,
    chunk_overlap=80,
    length_function=len,
)

all_chunks = []
all_metadatas = []

for file_path in data_dir.iterdir():
    if file_path.suffix.lower() in file_extensions and file_path.is_file():
        with file_path.open("r", encoding="utf-8") as f:
            text = f.read()

        chunks = text_splitter.split_text(text)
        print(f"Loaded {len(chunks)} chunks from {file_path.name}.")

        # For each chunk, create metadata with the source filename
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"source": file_path.name})

print(f"Total chunks from all documents: {len(all_chunks)}")

# 2. Set up embeddings and Chroma vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="notes_collection",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

ids = [f"chunk-{i}" for i in range(len(all_chunks))]
vector_store.add_texts(
    texts=all_chunks,
    metadatas=all_metadatas,
    ids=ids,
)

# 3. Create a retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 4. Set up the LLM (chat model)
llm = ChatOllama(model="llama3.2")

# 5. Create a prompt template that includes retrieved context
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant. Use the following context to answer the user's question.
If the context does not contain the answer, say you don't know and do not make things up.

Context:
{context}

Question:
{question}
"""
)

# 6. Chat loop with RAG
history: list = []

print("RAG chat is ready. Ask questions about your documents in the data folder. Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # 6a. Retrieve relevant docs for the question
    docs = retriever.invoke(user_input)
    context = "\n\n".join(doc.page_content for doc in docs)

    # 6b. Build the full prompt with context + question
    full_prompt = prompt.format(context=context, question=user_input)

    # 6c. Call the LLM with the constructed prompt
    response = llm.invoke(full_prompt)
    ai_reply = response.content

    print(f"\nAI:\n{ai_reply}")

    print("\n[Context used:]")
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        print(f"Source {i} (file: {src}):")
        print(doc.page_content)
        print("-" * 40)

    # 6d. (Optional) store history if you want to extend later
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=ai_reply))
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


# 1. Read and chunk the document (same as before)
with open("../data/notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=150,
    chunk_overlap=30,
    length_function=len,
)

chunks = text_splitter.split_text(text)
print(f"Loaded document with {len(chunks)} chunks.")

# 2. Set up embeddings and Chroma vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="notes_collection",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

# Add chunks (if running multiple times, Chroma will de-duplicate)
ids = [f"chunk-{i}" for i in range(len(chunks))]
vector_store.add_texts(texts=chunks, ids=ids)

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

print("RAG chat is ready. Ask questions about your notes.txt. Type 'exit' to quit.")

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
        print(f"Source {i}:")
        print(doc.page_content)
        print("-" * 40)

    # 6d. (Optional) store history if you want to extend later
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=ai_reply))
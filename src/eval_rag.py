from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# 1. Prepare RAG components (similar to rag_chat.py, but no loop)
with open("../data/notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
)
chunks = text_splitter.split_text(text)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="notes_collection_eval",
    embedding_function=embeddings,
    persist_directory="chroma_db_eval",
)

ids = [f"chunk-{i}" for i in range(len(chunks))]
vector_store.add_texts(texts=chunks, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant. Use the following context to answer the user's question.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question:
{question}
"""
)

# 2. Define a tiny evaluation set
test_cases = [
    {
        "question": "What is RAG?",
        "must_contain": ["Retrieval-Augmented Generation"],
    },
    {
        "question": "What is this notes file mainly about?",
        "must_contain": ["RAG", "project"],
    },
]

# 3. Run evaluation
results = []

for case in test_cases:
    question = case["question"]
    expected_snippets = case["must_contain"]

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    full_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(full_prompt)
    answer = response.content

    passed = all(snippet.lower() in answer.lower() for snippet in expected_snippets)

    results.append((question, passed, answer))

# 4. Print summary
print("Evaluation results:\n")
for question, passed, answer in results:
    status = "PASS" if passed else "FAIL"
    print(f"Q: {question}")
    print(f"Result: {status}")
    print(f"Answer: {answer}")
    print("-" * 60)
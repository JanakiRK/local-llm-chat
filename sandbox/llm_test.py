from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Define the model
model = ChatOllama(model="llama3.2")

# Define a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}")
])

# Create a chain
chain = prompt | model

# Run a simple query
response = chain.invoke({"user_input": "Explain what RAG (Retrieval-Augmented Generation) means in simple terms in one sentence."})
print("Model response:")
print(response.content)
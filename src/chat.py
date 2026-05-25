from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

model = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])

chain = prompt | model
history = []

print("Mini local ChatGPT is ready. Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    result = chain.invoke({"history": history, "user_input": user_input})
    ai_reply = result.content
    print(f"AI: {ai_reply}")

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=ai_reply))
# Memory and Chat Bots

from langchain import OpenAI, ConversationChain
import os
os.environ['OPENAI_API_KEY'] = "..."

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=False)

# conversation.predict(input="Hi there!")
# conversation.predict(input="Can we talk about the weather?")
# print(conversation.predict(input="It's a beautiful day!"))

print("Welcome to your AI Chatbot! What's on your mind?")
for _ in range(0, 3):
    human_input = input("You: ")
    ai_response = conversation.predict(input=human_input)
    print(f"AI: {ai_response}")
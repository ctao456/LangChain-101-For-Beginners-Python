# Storing and Retrieving Chat History

from langchain import OpenAI, ConversationChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
import os
os.environ['OPENAI_API_KEY'] = "..."

history = ChatMessageHistory()
history.add_user_message("Hello! Let's talk about giraffes.")
history.add_ai_message("Hi! I'm down to talk about giraffes.")

dicts = messages_to_dict(history.messages)
# print(dicts)

new_messages = messages_from_dict(dicts)

llm = OpenAI(temperature=0)
history = ChatMessageHistory(messages=new_messages)
buffer = ConversationBufferMemory(chat_memory=history)
conversation = ConversationChain(llm=llm,
                                 memory=buffer,
                                 verbose=True)

print(conversation.predict(input="What are they?"))
# print(conversation.memory)
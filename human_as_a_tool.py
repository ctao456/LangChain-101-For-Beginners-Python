from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
import os
os.environ['OPENAI_API_KEY'] = "..."

llm = OpenAI(temperature=0)
tools = load_tools(["human"])

agent_chain = initialize_agent(tools,
                               llm,
                               agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                               verbose=True)
agent_chain.run("What's my friend Andi's surname?")
# Action Agents

import pprint
from langchain.llms import OpenAI
from langchain.agents import get_all_tool_names, load_tools, initialize_agent
import os
os.environ['OPENAI_API_KEY'] = "..."

prompt = "When was the 3rd president of the United States born? What's that year raised to the power of 3?"

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(get_all_tool_names())

llm = OpenAI(temperature=0)
tools = load_tools(["wikipedia", "llm-math"],
                   llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run(prompt)
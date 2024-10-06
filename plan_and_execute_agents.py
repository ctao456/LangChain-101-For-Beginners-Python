# Plan and Execute Agent

# First plan the series of steps, and then execute each one

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute.agent_executor import PlanAndExecute
from langchain_experimental.plan_and_execute.executors.agent_executor import load_agent_executor
from langchain_experimental.plan_and_execute.planners.chat_planner import load_chat_planner
from langchain import SerpAPIWrapper, LLMMathChain, WikipediaAPIWrapper
from langchain.agents import Tool

import os
os.environ['OPENAI_API_KEY'] = "..."
os.environ['SERPAPI_API_KEY'] = "..."

llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm,
                                       verbose=True)
search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()

tools = [Tool(name="Search",
              func=search.run,
              description="useful for when you need to answer questions about current events"),
         Tool(name="Wikipedia",
              func=wikipedia.run,
              description="useful for when you need to look up facts and statistics"),
         Tool(name="Calculator",
              func=llm_math_chain.run,
              description="useful for when you need to answer questions about math")]

# Search for the location
# Identify the country
# Search for the population of the country
# Raise the population to .43
# Return
prompt = "Where are the next winter olympics going to be hosted? What is the population of that country raised to the 0.43 power?"

# planner and executor should use memory i.e. chat models to leverage previous information
model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run(prompt)
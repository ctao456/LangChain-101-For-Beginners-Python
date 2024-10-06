# Using different models with LangChain

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "..."

from langchain import HuggingFaceHub

llm = HuggingFaceHub(repo_id="google/flan-t5-base",
                     model_kwargs={"temperature": 0, "max_length": 64})

prompt = "What are good fitness tips?"

print(llm(prompt))
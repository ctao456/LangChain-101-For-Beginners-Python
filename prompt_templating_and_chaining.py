# Prompt templating and chaining

from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

import os
os.environ['OPENAI_API_KEY'] = "..."

template = "You are a naming consultant for new companies. What is a good name for a {company} that makes {product}?"

prompt = PromptTemplate.from_template(template)
print(prompt.format(company="ABC Startup", product="colorful socks"))

llm = OpenAI(temperature=0.9)

# LLM Chain: prompt + LLM
chain=LLMChain(llm=llm,
               prompt=prompt)
print(chain.run({'company': "ABC Startup",
                 'product': "colorful socks"}))
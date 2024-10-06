import os
os.environ['OPENAI_API_KEY'] = "..."

from langchain.llms import OpenAI

# Deafult OpenAI model is text-davinci-003
llm = OpenAI(temperature=0.9) # temperature determines randomness, lies between 0 and 1

prompt = "What would a good company name be for a company that makes colorful socks?"

result = llm.generate([prompt]*5)

for company_name in result.generations:
    print(company_name[0].text)
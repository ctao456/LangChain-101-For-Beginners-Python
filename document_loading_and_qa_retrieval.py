# Document loading and QA Retrieval

from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

import os
os.environ['OPENAI_API_KEY'] = "..."

loader = TextLoader('./state-of-the-union-23.txt')
documents = loader.load()
# print(documents)

# 3 levels of splitting: \n\n, \n, " "
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# print(texts[0], "**********************************\n")
# print(texts[1])

embeddings = OpenAIEmbeddings()
# Create vector database
store = Chroma.from_documents(texts, embeddings, collection_name="state-of-the-union")

llm = OpenAI(temperature=0)
chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())

print(chain.run("What did Biden talk about Ohio?"))


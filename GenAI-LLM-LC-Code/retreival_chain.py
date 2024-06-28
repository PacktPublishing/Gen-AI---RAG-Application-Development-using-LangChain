# This program is intended to demo the use of the following:
# 1. WebBaseLoader to read a webpage 
# 2. RecursiveCharacterTextSplitter to chunk the content into documents
# 3. Convert the documents into embeddings and store into an FAISS DB
# 4. Create a Stuff document chain, create a retrieval chain from the FAISS Db
# 5. Create a Retreival Chain using the FAISS retreiver and document chain

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

loader = WebBaseLoader("https://code4x.dev/courses/chat-app-using-langchain-openai-gpt-api-pinecone-vector-database/")

docs = loader.load()

# The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size. 
# It does this by using a set of characters. The default characters provided to it are ["\n\n", "\n", " ", ""].
text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

llm = ChatOpenAI()

embeddings = OpenAIEmbeddings()

# FAISS (Facebook AI Similarity Search) is a library that allows developers to store and search for embeddings of 
# documents that are similar to each other. 
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)   # document chain being part of the retrieval Chain

response = retrieval_chain.invoke({"context": "You are the trainer who is teaching the given course and you are to suggest to potential learners", 
                                   "input": "What are the key takeaways for learners from the Course?"})

print(response["answer"])


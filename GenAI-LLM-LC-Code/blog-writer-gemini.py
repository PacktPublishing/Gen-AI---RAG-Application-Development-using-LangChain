# This program is intended to Write Blogs by referring to a given webpage:
# This uses Google Gemini

from langchain_community.chat_models import ChatOpenAI
import google.generativeai as genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

loader = WebBaseLoader("https://medium.com/swlh/algorithmic-management-what-is-it-and-whats-next-33ad3429330b")

docs = loader.load()

# The RecursiveCharacterTextSplitter takes a large text and splits it based on a specified chunk size. 
# It does this by using a set of characters. The default characters provided to it are ["\n\n", "\n", " ", ""].
text_splitter = RecursiveCharacterTextSplitter()

documents = text_splitter.split_documents(docs)

#llm = ChatOpenAI(model_name="gpt-3.5-turbo")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

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

response = retrieval_chain.invoke(
    {"context": "You are a content writer who is creating a LinkedIn blog for Technology enthusiasts.", 
                                   "input": 
                                   """Please write a blog on the given content that sounds professional. 
                                      The blog should be more than 500 words and well structured in distict chapters.
                                      Use any facts, data or statistics available in the given input. 
                                   """,
                                   })


print(response["answer"])


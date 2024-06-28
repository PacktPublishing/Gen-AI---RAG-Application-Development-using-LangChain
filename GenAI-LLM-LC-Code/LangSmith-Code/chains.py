from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langsmith import traceable
from dotenv import load_dotenv
import os 

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langsmith_tracing = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["GOOGLE_API_KEY"] = api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = langsmith_tracing

# Set yout LLM to Google Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-pro",  google_api_key=os.getenv("GOOGLE_API_KEY"), convert_system_message_to_human=True)

def main():
    load_dotenv()
    simplechain()
    retreivalkchain()

@traceable(
    run_type="llm",
    name="OpenAI Call Decorator 1",
    tags=["simplechain"],
    metadata={"chainname": "simplechain"}
)
def simplechain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    print(chain.invoke({"input": "how can langsmith help with testing?"}))

@traceable(
    run_type="llm",
    name="OpenAI Call Decorator 2",
    tags=["retreivalkchain"],
    metadata={"chainname": "retreivalkchain"}
)
def retreivalkchain():
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)

    response = document_chain.invoke({
        "input": "how can langsmith help with testing?",
        "context": [Document(page_content="langsmith can let you visualize test results")]
    })

    print(response)
    print("#-----------#")
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
    print(response["answer"])


if __name__ == "__main__":
    main()
# This program takes in a Folder as Input (Hardcoded). This folder containes downloaded web pages from a Website.
# https://www.hrhelpboard.com/hr-policies.html  
# The folder has a sub and sub-sub-folder structure containing hundreds of .html pages that make up the given website.
# Our program recursively reads through the files (Pages/documents) in the folder, applies a CharacterTextSplitter to create chunks,
# loads these individual Chunks as one Vector into a local Vector Database - FAISS
# As a further experiment, we trigget a Query on this DB using Semantic Similarity Search which extracts and displays the 
# relevant chunks (Documents) retrieved by the Semantic Search.  

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS

def upload_htmls():
    """
    This function does the following:
    1. Reads recursively through the given folder hr-policies (withon current folder)
    2. Loads the Pages (Documents)   
    3. Loaded documents are split into chunks using Splitter 
    4. These chunks are converted into Language Embeddings and loaded as vectors into a local FAISS Vectos Database 
    """
    
    # Load all the HTML pages in the given folder structure recursively using Directory Loader
    loader = DirectoryLoader(path="hr-policies")
    documents = loader.load()
    print(f"{len(documents)} Pages Loaded")
    
    # Split loaded documents into Chunks using CharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]        
    )
    
    split_documents = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(split_documents)} Documents...")

    print(split_documents[0].metadata)

    # Upload chunks as vector embeddings into FAISS
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_documents, embeddings)
    # Save the FAISS DB locally
    db.save_local("faiss_index")

def faiss_query():
    """
    This function does the following:
    1. Load the local FAISS Database 
    2. Trigger a Semantic Similarity Search using a Query
    3. This retrieves semantically matching Vectors from the DB
    """
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings)

    query = "Explain the Candidate Onboarding process."
    docs = new_db.similarity_search(query)

    # Print all the extracted Vectors from the above Query
    for doc in docs:
        print("##---- Page ---##")
        print(doc.metadata['source'])
        print("##---- Content ---##")
        print(doc.page_content)

if __name__ == "__main__":
    # The below code 'upload_htmls()' is executed only once and then commented as the Vector Database is now built and ready for your further 
    # experiments
    # upload_htmls()   
    # The below function is experimental to trigger a semantic search on the Vector DB
    faiss_query()

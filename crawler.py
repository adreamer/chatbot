from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st
import os

pinecone_api_key = st.secrets.pinecone_api_key

os.environ["PINECONE_API_KEY"] = pinecone_api_key


import time


from langchain_community.document_loaders import PyPDFLoader

pdf_filepath = '전자금융거래법(법률)(제17354호)(20201210).pdf'
loader = PyPDFLoader(pdf_filepath)
docs = loader.load_and_split()

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 500,
#     chunk_overlap  = 100,
#     length_function = len,
# )

# texts = text_splitter.split_text(pages[0].page_content)

# len(texts)


from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)

# loader = TextLoader("../../how_to/state_of_the_union.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets.openai_api_key)

index_name = "langchain-index"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore(
    pinecone_api_key=pinecone_api_key,
    index_name=index_name,
    embedding=embeddings
)

docsearch = vectorstore.from_documents(docs, embeddings, index_name=index_name)


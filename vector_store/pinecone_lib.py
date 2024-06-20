import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_loaders import PyPDFLoader

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from llm.openai_lib import OpenAILib

import streamlit as st

import time

class PineconeLib:
    def __init__(self):
        self.pinecone_api_key = st.secrets.pinecone_api_key
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
        self.index_name = "langchain-index"  # change if desired
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.embeddings = OpenAILib().get_embeddings()
        self.compressor = None
        self.compression_retriever = None
        self.init_vectorstore()
        self.init_retriever()

    def init_retriever(self):
        self.retriever = self.vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 7, 'fetch_k': 50, 'lambda_mult': 0.2}
        )

    def init_compresser(self, llm):
        self.compressor = self.compressor or LLMChainExtractor.from_llm(llm)
        self.compression_retriever = self.compression_retriever or ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.retriever
        )

    def load_pdf(self, pdf_filepath):
        loader = PyPDFLoader(pdf_filepath)
        docs = loader.load_and_split()
        return docs

    def init_vectorstore(self):
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

        #index = pc.Index(index_name)

        self.vectorstore = PineconeVectorStore(
            pinecone_api_key=self.pinecone_api_key,
            index_name=self.index_name,
            embedding=self.embeddings
        )

        return self.vectorstore
    
    def store_docs(self, docs):
        docsearch = self.vectorstore.from_documents(docs, self.embeddings, index_name=self.index_name)

    def format_docs(self, docs):
        return '\n\n'.join([d.page_content for d in docs])

    def search(self, question):
        docs = self.retriever.invoke(question)
        print(len(docs))
        for doc in docs:
            print(doc.page_content)
        return docs
    
    def search_compressed(self, llm, question):
        self.init_compresser(llm)
        compressed_docs = self.compression_retriever.invoke(question)
        print(len(compressed_docs))
        for doc in compressed_docs:
            print(doc.page_content)
        return compressed_docs
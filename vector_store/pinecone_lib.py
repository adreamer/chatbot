import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_loaders import PyPDFLoader

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

import streamlit as st

import time

"""
    Pinecone용 벡터DB 라이브러리
    Pinecone에 저장하고 retrieve하는 메소드 제공
    
    ~/.streamlit/secrets.toml에 pinecone key 저장
    pinecone_api_key = ####
"""

class PineconeLib:
    def __init__(self, embeddings):
        self.pinecone_api_key = st.secrets.pinecone_api_key
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
        self.index_name = "langchain-index"  # change if desired
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.embeddings = embeddings
        self.compressor = None
        self.compression_retriever = None
        self.init_vectorstore()
        self.init_retriever()

    # 일반 Retriever 생성
    def init_retriever(self):
        self.retriever = self.vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 7, 'fetch_k': 50, 'lambda_mult': 0.2}
        )

    # LLM Compressed Retriever 생성
    def init_compresser(self, llm):
        self.compressor = self.compressor or LLMChainExtractor.from_llm(llm)
        self.compression_retriever = self.compression_retriever or ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.retriever
        )

    # PDF 로드
    def load_pdf(self, pdf_filepath):
        loader = PyPDFLoader(pdf_filepath)
        docs = loader.load_and_split()
        return docs

    def init_vectorstore(self):
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

        # Pinecone에 없으면 기본으로 생성
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # openai 임베딩 차원수임
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

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
import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

import time

"""
    Bedrock Knowledge Base용 라이브러리
    Bedrock Knowledge Base에서 retrieve하는 메소드 제공
    Knowledge Base에 저장하는 것은 콘솔에서 할 것
    
    AWS IAM key를 ~/.aws/credentials에 넣어야함
    [default]
    aws_access_key_id=####
    aws_secret_access_key=####
"""


class BedrockKbLib:
    def __init__(self):
        self.compressor = None
        self.compression_retriever = None
        self.init_retriever()

    # 일반 Retriever 생성
    def init_retriever(self):
        self.retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id="81V8DMRQZF",
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
            credentials_profile_name="sikim",
            region_name="us-west-2"
        )

    # LLM Compressed Retriever 생성
    def init_compresser(self, llm):
        self.compressor = self.compressor or LLMChainExtractor.from_llm(llm)
        self.compression_retriever = self.compression_retriever or ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.retriever
        )

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

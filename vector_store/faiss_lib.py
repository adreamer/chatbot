import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS
from llm.bedrock_lib import BedrockLib


class FAISSLib:
    def __init__(self):
        self.index_name = "langchain-index"  # change if desired
        self.embeddings = BedrockLib().get_embeddings()
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
        if os.path.exists(f"faiss_index/{self.index_name}.faiss"):
            self.vectorstore = FAISS.load_local("faiss_index", embeddings=self.embeddings, index_name=self.index_name,
                                                allow_dangerous_deserialization=True)
        else:
            self.vectorstore = FAISS.from_texts(["text"], embedding=self.embeddings)
            self.vectorstore.save_local("faiss_index", index_name=self.index_name)

        return self.vectorstore

    def store_docs(self, docs):
        docsearch = self.vectorstore.from_documents(docs, embedding=self.embeddings, index_name=self.index_name)
        self.vectorstore.save_local("faiss_index", index_name=self.index_name)

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

import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS

"""
    FAISS용 벡터DB 라이브러리
    FAISS에 저장하고 retrieve하는 메소드 제공
    기본으로 faiss_index 디렉토리에 파일로 관리하도록 함
"""
class FAISSLib:
    def __init__(self, embeddings):
        self.index_name = "langchain-index"  # change if desired
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

    # FAISS 내용은 로컬 파일에 관리
    def init_vectorstore(self):
        if os.path.exists(f"faiss_index/{self.index_name}.faiss"):
            self.vectorstore = FAISS.load_local("faiss_index", embeddings=self.embeddings, index_name=self.index_name,
                                                allow_dangerous_deserialization=True)
        else:
            self.vectorstore = FAISS.from_texts(["text"], embedding=self.embeddings)
            self.vectorstore.save_local("faiss_index", index_name=self.index_name)

        return self.vectorstore

    # Document를 FAISS에 저장
    def store_docs(self, docs):
        new_vectorstore = FAISS.from_documents(docs, embedding=self.embeddings)
        self.vectorstore.merge_from(new_vectorstore)
        self.vectorstore.save_local("faiss_index", index_name=self.index_name)

    def format_docs(self, docs):
        return '\n\n'.join([d.page_content for d in docs])

    # 일반 검색
    def search(self, question):
        docs = self.retriever.invoke(question)
        print(len(docs))
        for doc in docs:
            print(doc.page_content)
        return docs

    # LLM Compressed 검색
    def search_compressed(self, llm, question):
        self.init_compresser(llm)
        compressed_docs = self.compression_retriever.invoke(question)
        print(len(compressed_docs))
        for doc in compressed_docs:
            print(doc.page_content)
        return compressed_docs

    # DB에 저장된 목록 출력용
    def print_info(self):
        print(self.vectorstore.docstore._dict)
        print(f"총 count:{self.vectorstore.index.ntotal}")
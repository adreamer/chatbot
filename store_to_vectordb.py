import os

from vector_store.pinecone_lib import PineconeLib
from vector_store.faiss_lib import FAISSLib
from llm.openai_lib import OpenAILib
from llm.bedrock_lib import BedrockLib

"""
    문서를 Vector DB에 ingest하는 스크립트
"""

llm = BedrockLib()
vectorstore_lib = FAISSLib(llm.get_embeddings())

# 개별 파일 저장
def load_file():
    pdf_filepath = 'docs/금융분야 클라우드컴퓨팅서비스 이용 가이드(부분개정)_FN.pdf'
    docs = vectorstore_lib.load_pdf(pdf_filepath)
    vectorstore_lib.store_docs(docs)

# 디렉토리에 있는 파일 저장
def load_directory():
    docs_dir = "docs"

    for file_name in os.listdir(docs_dir):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(docs_dir, file_name)
            if os.path.isfile(file_path):
                docs = vectorstore_lib.load_pdf(file_path)
                vectorstore_lib.store_docs(docs)

load_directory()
import os

from vector_store.pinecone_lib import PineconeLib
from vector_store.faiss_lib import FAISSLib

vectorstore_lib = FAISSLib()


def load_file():
    pdf_filepath = 'docs/금융분야 클라우드컴퓨팅서비스 이용 가이드(부분개정)_FN.pdf'
    docs = vectorstore_lib.load_pdf(pdf_filepath)
    vectorstore_lib.store_docs(docs)


def load_directory():
    docs_dir = "docs"

    for file_name in os.listdir(docs_dir):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(docs_dir, file_name)
            if os.path.isfile(file_path):
                docs = vectorstore_lib.load_pdf(file_path)
                vectorstore_lib.store_docs(docs)

load_directory()
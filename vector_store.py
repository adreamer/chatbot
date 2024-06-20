from pinecone_lib import PineconeLib

pdf_filepath = '금융분야 클라우드컴퓨팅서비스 이용 가이드(부분개정)_FN.pdf'

vectorstore_lib = PineconeLib()
docs = vectorstore_lib.load_pdf(pdf_filepath)
vectorstore_lib.store_docs(docs)
from pinecone_lib import PineconeLib

pdf_filepath = '금융분야 클라우드컴퓨팅서비스 이용 가이드(부분개정)_FN.pdf'

vectorstore_lib = PineconeLib()
docs = vectorstore_lib.load_pdf(pdf_filepath)
vectorstore_lib.store_docs(docs)

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 500,
#     chunk_overlap  = 100,
#     length_function = len,
# )

# texts = text_splitter.split_text(pages[0].page_content)

# len(texts)

# loader = TextLoader("../../how_to/state_of_the_union.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)


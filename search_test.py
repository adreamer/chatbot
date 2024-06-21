from vector_store.pinecone_lib import PineconeLib
from vector_store.faiss_lib import FAISSLib
from llm.openai_lib import OpenAILib
from llm.bedrock_lib import BedrockLib

llm_lib = BedrockLib()
llm = llm_lib.get_llm()
vectorstore_lib = FAISSLib(llm_lib.get_embeddings())

vectorstore_lib.print_info()

prompt = input("Search? ")
print("==== search ====")
vectorstore_lib.search(prompt)

print("==== compressed ====")
vectorstore_lib.search_compressed(llm, prompt)
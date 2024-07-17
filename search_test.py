from vector_store.pinecone_lib import PineconeLib
from vector_store.faiss_lib import FAISSLib
from vector_store.bedrockkb_lib import BedrockKbLib
from llm.openai_lib import OpenAILib
from llm.bedrock_lib import BedrockLib

"""
    Vector Store 검색 테스트용
    기본 검색 결과와 LLM으로 압축된 결과를 보여줌
"""

llm_lib = BedrockLib()
llm = llm_lib.get_llm()
vectorstore_lib = BedrockKbLib()

prompt = input("Search? ")

# 일반 검색
print("==== search ====")
vectorstore_lib.search(prompt)

# LLM Compressed 결과
print("==== compressed ====")
vectorstore_lib.search_compressed(llm, prompt)
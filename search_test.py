from pinecone_lib import PineconeLib
from openai_lib import OpenAILib

vectorstore_lib = PineconeLib()
llm = OpenAILib().get_llm()

prompt = input("Search?")
print("==== search ====")
docs = vectorstore_lib.search(prompt)

print("==== compressed ====")
docs = vectorstore_lib.search_compressed(llm, prompt)
from langchain_aws import ChatBedrock, BedrockEmbeddings


class BedrockLib:
    def __init__(self):
        return

    def get_llm(self, model_name="claude-3-sonnet-20240229", temperature=0.0):
        llm = ChatBedrock(
            model_id=model_name,
            model_kwargs={"temperature": temperature}
        )
        return llm

    def get_embeddings(self, model_name="amazon.titan-embed-text-v1"):
        embeddings = BedrockEmbeddings(
            model_id=model_name,
            region_name="us-east-1"
        )
        return embeddings

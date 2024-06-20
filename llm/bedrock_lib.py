from langchain_aws import ChatBedrock, BedrockEmbeddings


class BedrockLib:
    def __init__(self):
        return

    def get_llm(self, model_name="anthropic.claude-3-sonnet-20240229-v1:0", temperature=0.0):
        llm = ChatBedrock(
            model_id=model_name,
            model_kwargs={"temperature": temperature},
            region_name="us-east-1"
        )
        return llm

    def get_embeddings(self, model_name="amazon.titan-embed-text-v2:0"):
        embeddings = BedrockEmbeddings(
            model_id=model_name,
            region_name="us-east-1"
        )
        return embeddings

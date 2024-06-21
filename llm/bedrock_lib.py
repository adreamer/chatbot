from langchain_aws import ChatBedrock, BedrockEmbeddings

"""
    Bedrock용 LLM, 임베딩 라이브러리
    기본적으로 LLM은 Anthropic Claude 3.5 Sonnet, 임베딩은 Amazon Titan
    
    AWS IAM key를 ~/.aws/credentials에 넣어야함
    [default]
    aws_access_key_id=####
    aws_secret_access_key=####
"""
class BedrockLib:
    def __init__(self):
        return

    def get_llm(self, model_name="anthropic.claude-3-5-sonnet-20240620-v1:0", temperature=0.0):
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

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st

"""
    OpenAI용 LLM, 임베딩 라이브러리
    기본적으로 LLM은 GPT-4o, 임베딩은 text-embedding-3-small
"""
class OpenAILib:
    def __init__(self):
        # Get an OpenAI API Key before continuing
        if "openai_api_key" in st.secrets:
            self.openai_api_key = st.secrets.openai_api_key
        else:
            self.openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if not self.openai_api_key:
            st.info("Enter an OpenAI API Key to continue")
            st.stop()

    def get_llm(self, model_name="gpt-4o", temperature=0.0):
        llm = ChatOpenAI(api_key=self.openai_api_key, model=model_name, temperature=temperature)
        return llm

    def get_embeddings(self, model_name="text-embedding-3-small"):
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=model_name)
        return embeddings

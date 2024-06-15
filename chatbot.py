import os
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_anthropic import ChatAnthropic

import streamlit as st

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

"""
A basic example of using StreamlitChatMessageHistory to help LLMChain remember messages in a conversation.
The messages are stored in Session State across re-runs automatically. You can view the contents of Session State
in the expander below. View the
[source code for this app](https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py).
"""

pinecone_api_key = st.secrets.pinecone_api_key

os.environ["PINECONE_API_KEY"] = pinecone_api_key


from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Set up Vectorstore
def get_vectorstore():
    pc = Pinecone(api_key=pinecone_api_key)

    index_name = "langchain-index"  # change if desired
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets.openai_api_key)

    vectorstore = PineconeVectorStore(
        pinecone_api_key=pinecone_api_key,
        index_name=index_name,
        embedding=embeddings
    )
    return vectorstore

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

def search(llm, vectorstore, question):
    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'fetch_k': 20, 'lambda_mult': 0.4}
    )

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    compressed_docs = compression_retriever.get_relevant_documents(question)
    print(len(compressed_docs))
    print(compressed_docs[0].page_content)
    return compressed_docs


vectorstore = get_vectorstore()

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# Get an OpenAI API Key before continuing
if "openai_api_key" in st.secrets:
    openai_api_key = st.secrets.openai_api_key
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

# Set up the LangChain, passing in Message History

template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''
human_template = HumanMessagePromptTemplate.from_template(template)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an professional in law and regulations about finanace and technology."),
        MessagesPlaceholder(variable_name="history"),
        human_template,
    ]
)

#os.environ["ANTHROPIC_API_KEY"] = st.secrets.antrhopic_api_key
#chain = prompt | ChatAnthropic(model="claude-3-sonnet-20240229")
llm = ChatOpenAI(api_key=openai_api_key)
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "any"}}
    docs = search(llm, vectorstore, prompt)
    response = chain_with_history.invoke({"question": prompt, "context": format_docs(docs)}, config)
    st.chat_message("ai").write(response.content)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)

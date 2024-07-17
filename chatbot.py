from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

import streamlit as st

#    메인 챗봇 스크립트
#    streamlit run chatbot.py 로 실행

SELECTED_STORE = "BEDROCK"  # FAISS or PINECONE or BEDROCK
USE_BEDROCK = True  # Bedrock or OpenAI
USE_COMPRESSED_SEARCH = False  # LLM Compressed or 전체 검색 결과


# LLM 생성
if USE_BEDROCK:
    from llm.bedrock_lib import BedrockLib
    llm_lib = BedrockLib()

else:
    from llm.openai_lib import OpenAILib
    llm_lib = OpenAILib()

llm = llm_lib.get_llm()

# 백터DB 생성
if SELECTED_STORE == "FAISE":
    from vector_store.faiss_lib import FAISSLib
    vectorstore_lib = FAISSLib(llm_lib.get_embeddings())
elif SELECTED_STORE == "PINECONE":
    from vector_store.pinecone_lib import PineconeLib
    vectorstore_lib = PineconeLib(llm_lib.get_embeddings())
else:
    from vector_store.bedrockkb_lib import BedrockKbLib
    vectorstore_lib = BedrockKbLib()


st.set_page_config(page_title="전자금융업 챗봇", page_icon="📖")
st.title("📖 전자금융업 챗봇")

# 채팅 기록용 메모리 생성
msgs = StreamlitChatMessageHistory(key="langchain_messages")
# OpenAI는 AI 첫 메세지 되지만 앤트로픽은 안됨
# if len(msgs.messages) == 0:
#     msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# RAG용 템플릿
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''
human_template = HumanMessagePromptTemplate.from_template(template)

# Chat history와 RAG Context 들어간 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an professional in law and regulations about finance and technology."),
        MessagesPlaceholder(variable_name="history"),
        human_template,
    ]
)

chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# 채팅 기록에 있는 메세지 출력
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "any"}}

    if USE_COMPRESSED_SEARCH:
        # LLM 압축된 결과 받아오기
        docs = vectorstore_lib.search_compressed(llm, prompt)
    else:
        # 검색된 모든 문서 받아오기
        docs = vectorstore_lib.search(prompt)

    # LLM에 프롬프트로 물어보기
    response = chain_with_history.invoke({"question": prompt, "context": vectorstore_lib.format_docs(docs)}, config)
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

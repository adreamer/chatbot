from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

# 백터DB
from vector_store.pinecone_lib import PineconeLib

# LLM
from llm.openai_lib import OpenAILib
from llm.bedrock_lib import BedrockLib

import streamlit as st

st.set_page_config(page_title="전자금융업 챗봇", page_icon="📖")
st.title("📖 전자금융업 챗봇")

# LLM 생성
#llm = OpenAILib().get_llm()
llm = BedrockLib().get_llm()

# 백터DB 생성
vectorstore_lib = PineconeLib()

# 채팅 기록용 메모리 생성
msgs = StreamlitChatMessageHistory(key="langchain_messages")
# if len(msgs.messages) == 0:
#     msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# RAG용 템플릿
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

    # LLM 압축된 결과 받아오기
    docs = vectorstore_lib.search_compressed(llm, prompt)

    # 검색된 모든 문서 받아오기
    #docs = vectorstore_lib.search(prompt)

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

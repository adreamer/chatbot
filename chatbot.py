from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from pinecone_lib import PineconeLib
from openai_lib import OpenAILib

#from langchain_anthropic import ChatAnthropic

import streamlit as st

st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

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
llm = OpenAILib().get_llm()
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

vectorstore_lib = PineconeLib()

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "any"}}
    docs = vectorstore_lib.search_compressed(llm, prompt)
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

# 此版本不使用Agent，强制直接使用retriever提供参考，与用户query一起输入到LLM中生成结果

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import numpy as np

# new version
#define the GPU for embedding

#自动分割代码段
# 1. 加载txt文件
with open(r"D:\coding\vector-cursor\cleaned_scl_codes2.txt", "r", encoding='utf-8') as f:
      scl_code = f.read()

# scl_code = TextLoader.load_and_split("combined_scl_codes.txt")

# 2. 代码分块（按函数/逻辑块分割）
text_splitter = RecursiveCharacterTextSplitter(
    separators='//',
    chunk_size=1000,
    chunk_overlap = 0
)
chunks = text_splitter.split_text(scl_code)
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_texts(chunks, embeddings)

print("*******************************************************\n")
print('EMBEDDING FINISHED')

from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.llms import huggingface_hub
from langchain.chains import LLMChain
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from langchain_ollama import OllamaLLM

st.set_page_config(page_title="Winder_SCL_Generator", layout="wide")
st.title("Winder SCL Generator")
retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.2},
)

msgs = StreamlitChatMessageHistory()
if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! I am the Winder SCL Generator Assistant. How can I help you today?"}]
# for msg in st.session_state.messages:
#     st.chat_message([msg["role"]]).write(msg["content"])
from langchain.tools.retriever import create_retriever_tool

memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output"
)

#指令模板
instructions ="""You are an agent designed to generate SCL for the Winder.
Meanwhile the user quesion is input, the SCL reference will also be given.
Your answer muset be based on the SCL reference.
"""


# Modify the base_prompt_template to be more strict
base_prompt_template ="""
{instructions}

TOOLS:
-----------------------------

You have access to the following tools:
{tools}

IMPORTANT: You MUST generate the conclusion in ONE response without ANY ITERATION and following the format:

1. Start with "Thought: [your reasoning]"
2. Then analyze the reference
3. Conclude the result get from the retriever
     Thought:[result]
     Final Answer: [your conclusion]
4. End with "Final Answer: [your conclusion]"

Previous conversation history:
{chat_history}

New input: {input}

{agent_scratchpad}
"""

base_prompt = PromptTemplate.from_template(base_prompt_template)
prompt = base_prompt.partial(instructions = instructions)

# 选择使用的大模型
# llm = OllamaLLM(model="deepseek-r1:7b",
#                 temperature= 0)
llm = OllamaLLM(model="llama3.1",
                temperature=1)
chain = LLMChain(promt = prompt,
                 llm = llm)
user_query = st.chat_input(placeholder='please issue an order')

if user_query:
    st.session_state.messages.append({'role':'user','content':user_query})
    st.chat_message('user').write(user_query)

    with st.chat_message("assistat"):
        st_cb = StreamlitCallbackHandler(st.container())
        config = {'callbacks':[st_cb]}

        docs = retriever.get_relevant_documents(user_query)
        finalQuery = "The question of user is:" +user_query + "/n The following is some useful references" + docs

        response = chain.invoke({
             finalQuery
        })

        st.session_state.messages.append({"role":"assistat","content":response["output"]})
        st.write(response["output"])
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import numpy as np

#define the GPU for embedding

#自动分割代码段
# 1. 加载txt文件
with open("42#.txt", "r") as f:
      scl_code = f.read()

# scl_code = TextLoader.load_and_split("combined_scl_codes.txt")

# 2. 代码分块（按函数/逻辑块分割）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap = 20
)
chunks = text_splitter.split_text(scl_code)

# embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from subprocess import Popen
from langchain_ollama import OllamaLLM
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

st.set_page_config(page_title="Winder_SCL_Generator", layout="wide")
st.title("Winder SCL Generator")
retriever = db.as_retriever(
         search_type="similarity_score_threshold",
         search_kwargs={"k": 5, "score_threshold": 0.2},
)

msgs = StreamlitChatMessageHistory()
if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! I am the Winder SCL Generator Assistant. How can I help you today?"}]
for msg in st.session_state.messages:
    st.chat_message([msg["role"]]).write(msg["content"])
from langchain.tools.retriever import create_retriever_tool
tool = create_retriever_tool(
     
    name="retrieverSSSSSS",
    description="SSSSSS_company_rules_Searcher",
    retriever=retriever,
)
tools = [tool]

memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output"
)

#指令模板
instructions = """You are an designed to search infoamation about the rules of SSSSSS company.
Your answer MUST be based on the context searched from the retriever.
If the retriever returned nothing, you can just return"Sry, I can't do that,plaese descripe it more sepecifically!"
"""

# 有聊天记录
# base_prompt_template ="""
# {instructions}


# '''
# *******************************************************
# Thought: Do I need to use the result from the retriever? Yes
# Context from the retriever:  {context}
# Question:{question}
# Observation: the result of the action
# '''

# Previous conversation history:
# {chat_history}

# """

# 没有聊天记录
base_prompt_template ="""
{instructions}


'''
*******************************************************
Thought: Do I need to use the result from the retriever? Yes
Context from the retriever:  {context}
Question:{question}
Observation: the result of the action
'''


"""

base_prompt = PromptTemplate.from_template(base_prompt_template)
prompt = base_prompt.partial(instructions = instructions)

# llm = OllamaLLM(model="deepseek-r1:7b")
llm = OllamaLLM(model="llama3.1")
user_query = st.chat_input(placeholder='please issue an order')

# if user_query:
#     st.session_state.messages.append({'role':'user','content':user_query})
#     st.chat_message('user').write(user_query)

#     with st.chat_message("assistat"):
#         st_cb = StreamlitCallbackHandler(st.container())
#         config = {'callbacks':[st_cb]}
#         response = agent_executor.invoke({'input':user_query},config=config)
#         st.session_state.messages.append({"role":"assistat","content":response["output"]})
#         st.write(response["output"])

# test without UI
user_query = 'i am a junior administrator, how much is the salary in SSSSSS company?'
if user_query:

    retrieved_docs = retriever.invoke(user_query)
    if not retrieved_docs:
        print("No relevant context found in the document to answer your question.")    

    formatted_input = {
        "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
        "question": user_query,
    }

    # Build the RAG chain
    chain = (
        RunnablePassthrough()  # Passes the input as-is
        | prompt           # Formats the input for the LLM
        | llm            # Queries the LLM
        | StrOutputParser()     # Parses the LLM's output
    )

    response = chain.invoke(formatted_input)

    dic = dict(input=user_query)
    # response = agent_executor.invoke(dic)
    print(response)
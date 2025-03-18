from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import numpy as np
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
import os

st.set_page_config(page_title="Winder_SCL_Generator", layout="wide")
st.title("Winder SCL Generator")

# Load and split text files
input_directory = 'D:/coding/vector/ori_converting_scl_codes'
scl_code = ""
for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):
        with open(os.path.join(input_directory, filename), 'r') as f:
            scl_code += f.read() + "\n"  # Add a newline to separate contents of different files

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = text_splitter.split_text(scl_code)

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_texts(chunks, embeddings)

retriever = db.as_retriever()
msgs = StreamlitChatMessageHistory()
if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! I am the Winder SCL Generator Assistant. How can I help you today?"}]

from langchain.tools.retriever import create_retriever_tool
tool = create_retriever_tool(
    name="retriever",
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

instructions = """You are an agent designed to search information about the rules of SSSSSS company.
You MUST use the retriever to search for the information and answer the questions by it.
Maybe you can answer the questions without searching, but still you MUST search with the retriever.
If you can't find the relevant information in the retriever, you can just return "Sorry, I can't do that, please describe it more specifically!"
"""

base_prompt_template = """
{instructions}

TOOLS:
tool, retriever

You have access to the following tools:
{tools}

Any questions about SSSSSS company, you must use the tools.

To use a tool, please use the following format:
ZWJ'''
*******************************************************
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: {input}
Observation: the result of the action
'''

If you do not need to use a tool, you MUST use the format:
ZWJ'''
*******************************************************
Thought: Do I need to use a tool? No
Final Answer: [your response here]
'''

Begin!
{agent_scratchpad}
Previous conversation history:
{chat_history}

New input:
{input}
"""

base_prompt = PromptTemplate.from_template(base_prompt_template)
prompt = base_prompt.partial(instructions=instructions)

llm = OllamaLLM(model="deepseek-r1:7b")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                               handle_parsing_errors='No matching contents')

user_query = st.chat_input(placeholder='please issue an order')

# Test without UI
user_query = 'SSSSSS company rule, the basic salary of a junior administrative position is?'
if user_query:
    dic = dict(input=user_query)
    response = agent_executor.invoke({"input": user_query})
    print(response)
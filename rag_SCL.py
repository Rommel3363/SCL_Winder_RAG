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
with open(r"D:\coding\vector-cursor\cleaned_scl_codes.txt", "r", encoding='utf-8') as f:
      scl_code = f.read()

# scl_code = TextLoader.load_and_split("combined_scl_codes.txt")

# 2. 代码分块（按函数/逻辑块分割）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap = 10
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

st.set_page_config(page_title="Winder_SCL_Generator", layout="wide")
st.title("Winder SCL Generator")
retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.2},
)

msgs = StreamlitChatMessageHistory()
if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! I am the Winder SCL Generator Assistant. How can I help you today?"}]
# for msg in st.session_state.messages:
#     st.chat_message([msg["role"]]).write(msg["content"])
from langchain.tools.retriever import create_retriever_tool
# tool = create_retriever_tool(
#     name="retrieverSSSSSS",
#     description="SSSSSS_company_rules_Searcher",
#     retriever=retriever,
# )

tool = create_retriever_tool(
    retriever=retriever,
    name="retriever",
    description="Winder Scl Assistant",
)
tools = [tool]
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output"
)

#指令模板
instructions ="""You are an angent designed to generate SCL for the Winder.
You can use the retriever to search for the information and answer the questions by it.
Maybe you can answer the questions without searching, but still you must search with the retriever.
If you can't find the relevant information in the retriever, you can just return"Sry, I can't do that,plaese descripe it more sepecifically!"
"""
base_prompt_template ="""
{instructions}

TOOLS:
-----------------------------

You have access to the following tools:
{tools}

Any questions about SCL code, you must use the tools.

To use a tool, you MUST use the exact format:

Thought: [your reasoning about whether to use a tool]
Action: [the action to take, must be one of: {tool_names}]
Action Input: [the input to the tool]
Observation: [the result of the action]
Thought: [your reasoning about the observation]
Action: [next action if needed]
... (repeat as needed)
Final Answer: [your final response]

If you do not need to use a tool, you MUST use this exact format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Remember: ALWAYS follow these exact formats. Each response must start with "Thought:" and end with a "Final Answer:".

Begin!

Previous conversation history:
{chat_history}

New input: {input}

{agent_scratchpad}
"""

base_prompt = PromptTemplate.from_template(base_prompt_template)
prompt = base_prompt.partial(instructions = instructions)


# tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
# llm = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-1b")


#测试是否是模型的影响，即是否是deepseek的模型不支持用agent的retriever
# from langchain.chat_models import init_chat_model

# model = init_chat_model("llama3-8b-8192", model_provider="groq")

# llm = OllamaLLM(model="deepseek-r1:7b")
llm = OllamaLLM(model="llama3.1")
agent = create_react_agent(llm,tools,prompt)

# 测试tool配置正确
# result = tool.run('SSSSSS company rule, the basic salary of a junior administrative position is?')
# print('tool的运行结果',result)
# print('工具列表:',[tool.name])

# no marching contents可能是来自于这里的
# agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
#                                handle_parsing_errors='No marching contents')
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    memory=memory, 
    verbose=True,
    handle_parsing_errors=True,
    # handle_parsing_errors="Check the format of your response. Make sure to include 'Thought:', 'Action:', 'Action Input:', and 'Final Answer:' in the correct order.",
    max_iterations=5  # 添加最大迭代次数限制
)

# user_query = st.chat_input(placeholder='please issue an order')

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
user_query = 'please write a SCL code to set the friction measurement curve'
# user_query = 'hello, who are you'
if user_query:
    dic = dict(input=user_query)
    # response = agent_executor.invoke(dic)
    response = agent_executor.invoke({"input":user_query})
    print(response)
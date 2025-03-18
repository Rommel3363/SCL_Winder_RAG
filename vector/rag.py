from langchain_community.vectorstores import FAISS
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

# ctx = get_script_run_ctx()
# process = Popen(['python','my_script.py'])
# add_script_run_ctx(process, ctx)


# try:
#     torch.set_default_device('cuda')
# except 'HA?':
#     print('wtf')
# else:
#     print('yeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeah!!!')

## angent 131,140
Embeddings = HuggingFaceEmbeddings()
db = FAISS.load_local(r"D:\coding\vector\scl_winding_db",embeddings=Embeddings)

st.set_page_config(page_title="Winder_SCL_Generator", layout="wide")
st.title("Winder SCL Generator")


retriever = db.as_retriever()


msgs = StreamlitChatMessageHistory()
if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! I am the Winder SCL Generator Assistant. How can I help you today?"}]
# for msg in st.session_state.messages:
#     st.chat_message([msg["role"]]).write(msg["content"])
from langchain.tools.retriever import create_retriever_tool
tool = create_retriever_tool(
    name="retriever",
    description="Search the scl database for relevant information",
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
instructions = """You are an angent designed to generate SCL for the Winder.
You can use the retriever to search for the information and answer the questions by it.
Maybe you can answer the questions without searching, but still you must search with the retriever.
If you can't find the relevant information in the retriever, you can just return"Sry, I can't do that,plaese descripe it more sepecifically!"
"""

base_prompt_template ="""
{instructions}

TOOLS:
-----

You have access to the following tools:
{tools}

To use a tool, please use the following format:
ZWJ'''
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input:{input}
Observation: the result of the action
ZWJ'''

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
ZWJ'''
Thought: Do I need to use a tool? No
Final Answer:[your response here]
ZWJ'''

Begin!
{agent_scratchpad}
Previous conversation history:
{chat_history}

New input:
{input}

"""

base_prompt = PromptTemplate.from_template(base_prompt_template)
prompt = base_prompt.partial(instructions = instructions)


tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
llm = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-1b")

agent = create_react_agent(llm,tools,prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                               handle_parsing_errors='No marching contents')

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
user_query = 'please generate a code for tension control mode'
if user_query:
    print(type(user_query))
    dic = dict(input=user_query)
    # response = agent_executor.invoke(dic)
    response = agent_executor.invoke({"input":user_query})
    # dic = dict(input=user_query)
    # response = agent_executor.invoke(input=dic)
    print(response)
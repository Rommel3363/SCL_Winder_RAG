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
    chunk_size=200,
    chunk_overlap = 50
)
chunks = text_splitter.split_text(scl_code)

# embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
embeddings = HuggingFaceEmbeddings()

db = FAISS.from_texts(chunks, embeddings)

retriever = db.as_retriever()
docs = retriever.get_relevant_documents(" the basic salary of a junior administrative position is")

print(docs)
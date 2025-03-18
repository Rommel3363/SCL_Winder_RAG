GPU = 'CUDA'

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import numpy as np

#define the GPU for embedding

#自动分割代码段
# 1. 加载SCL代码文件7
# with open("combined_scl_codes.txt", "r") as f:
#     scl_code = f.read()

with open("42#.txt", "r") as f:
      scl_code = f.read()

# scl_code = TextLoader.load_and_split("combined_scl_codes.txt")

# 2. 代码分块（按函数/逻辑块分割）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    # separators=["\nEND_REGION", "\nREGION_DESCRIPTION", "\n//"]
)
chunks = text_splitter.split_text(scl_code)




# print("*******************************************************\n")
# numpylist = np.array(chunks)
# print(type(numpylist))
# print(numpylist.shape)
# print(numpylist.ndim)

# 3. 生成嵌入并构建向量库
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings()

db = FAISS.from_texts(chunks, embeddings)
# db = embeddings.embed_query(chunks)
# db = embeddings.embed_documents(chunks)
db.save_local("42#")  # 保存到本地

print("*******************************************************\n")
# numpylist = np.array(chunks)
print('FINISHED')
# from langchain_openai import OpenAIEmbeddings

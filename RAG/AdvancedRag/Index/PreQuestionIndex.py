import os
import uuid
from pydantic import BaseModel, Field

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

docs = TextLoader("../../../文件/deepseek百度百科.txt", encoding='utf-8').load()
splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30, separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""])
chunks = splitter.split_documents(docs)

# 为文档生成id
idKey = 'doc_id'
ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

# 创建数据库
docStore = InMemoryStore()

# 不使用本地bge-m3的原因：会算出来NaN
embModel = DashScopeEmbeddings(model="text-embedding-v4")

vectorStore = Chroma(
    collection_name="ds_introduction",
    embedding_function=embModel
)
retriever = MultiVectorRetriever(
    docstore=docStore,
    vectorstore=vectorStore,
    id_key=idKey
)

print("正在将原始文档写入...")
retriever.docstore.mset(list(zip(ids, chunks)))
print("原始文档写入完成")

# 为每个文档写入预设性问题
print("正在生成预设性问题...")
llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)

preQuestion_prompt = ChatPromptTemplate.from_template(
    "为以下的文档生成三个问题，以便于搜索相关问题时可以更容易搜索到该文档\n"
    "文档内容:\n{doc}"
)

class Question(BaseModel):
    questions: list[str] = Field(..., description="根据文档生成的问题")

preQuestion_chain = (preQuestion_prompt|
                     llm.with_structured_output(Question)|
                     RunnableLambda(lambda x: x.questions))

# TODO：这里写得不是很好，有时间可以再优化一下
preQuestions = preQuestion_chain.batch(chunks, {"max_concurrency": 5})
print("预设性问题生成完成")

print("正在写入向量数据库...")
questions_docs = list()
for i, questions in enumerate(preQuestions):
    questions_docs.extend(
        [Document(metadata={idKey: ids[i]}, page_content=q) for q in questions]
    )

retriever.vectorstore.add_documents(questions_docs)
print("向量数据库写入完成")

final_prompt = ChatPromptTemplate.from_template(
    "根据以下文档回答用户问题，如果文档中没有提及，请回答我不知道\n"
    "文档内容:\n{doc}"
    "用户提问:\n{question}"
)

answer_chain = (
    RunnablePassthrough.assign(
        doc = RunnableLambda(lambda x: retriever.invoke(x["question"])) | RunnableLambda(lambda x: '\n\n'.join([d.page_content for d in x]))
    )|
    final_prompt |
    llm | StrOutputParser()
)

while True:
    user_data = input("请输入你的问题，或者输入stop退出: ")
    if user_data == "stop":
        break
    resp = answer_chain.invoke({"question" : user_data})
    print("大模型回复: ")
    print(resp)

print("程序结束")
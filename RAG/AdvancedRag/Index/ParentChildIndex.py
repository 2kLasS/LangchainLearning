import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

print("正在读取文档...")
docs = TextLoader("../../../文件/deepseek百度百科.txt", encoding='utf-8').load()
print("文档加载完成")

print("正在创建数据库与检索器...")
embModel = OllamaEmbeddings(model="bge-m3")
vectorStore = Chroma(
    collection_name="ds_introduction",
    embedding_function=embModel
)

docStore = InMemoryStore()

# 该类会自动为文档添加id
retriever = ParentDocumentRetriever(
    vectorstore=vectorStore,
    docstore=docStore,
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0),
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=0)
)
print("创建数据库与检索器成功")

print("正在嵌入文档...")
retriever.add_documents(docs)
print("嵌入文档成功...")

# result = retriever.invoke("deepseek成立的地点")
prompt = ChatPromptTemplate.from_template(
    "根据以下文档回答问题，如果文档没有提及，请回复我不知道:\n"
    "文档内容:\n"
    "{doc}\n"
    "用户提问:\n"
    "{question}"
)

llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)

chain = RunnablePassthrough.assign(
    doc = RunnableLambda(lambda x : retriever.invoke(x["question"])) | RunnableLambda(lambda x : '\n'.join([d.page_content for d in x]))
) | prompt | llm | StrOutputParser()

while True:
    user_data = input("请输入你的问题，或者输入stop退出: ")
    if user_data == "stop":
        break
    resp = chain.invoke({"question" : user_data})
    print("大模型回复: ")
    print(resp)

print("程序结束")

import os

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_content(docs):
    return "\n\n".join([f"{doc.metadata['page_content']}" for doc in docs])

# 读取文档
loader = UnstructuredWordDocumentLoader("../文件/人事管理流程.docx")
orgin_doc = loader.load()

# 切分文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
)
chunks = splitter.split_documents(orgin_doc)

# 嵌入模型+构建数据库
emb_model = OllamaEmbeddings(model="bge-m3")
vs = Chroma.from_documents(
    documents=chunks,
    embedding=emb_model,
    collection_name="work",
)
retriever = vs.as_retriever()

llm = ChatOpenAI(model="qwen3-max-2026-01-23", api_key=os.environ.get("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

prompt = ChatPromptTemplate.from_messages([
    SystemMessage("你是一个人工智能助手"),
    HumanMessagePromptTemplate.from_template("""
        根据以下内容进行回复，如果文档中没有提及，请回复"我不知道"
        {docs}
        
        问题: {question}
    """)
])

chain = prompt | llm | StrOutputParser()


while True:
    user_input = input("请输入你的问题, 输入stop结束: ")
    if user_input == "stop":
        break
    rag_result = retriever.invoke(user_input)
    rag_result = "\n\n".join([doc.page_content for doc in rag_result])
    resp = chain.invoke({"question": user_input, "docs": rag_result})
    for chunk in resp:
        print(chunk, end="", flush=True)
    print()
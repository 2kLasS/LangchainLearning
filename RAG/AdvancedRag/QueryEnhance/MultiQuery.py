import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

docs = TextLoader("../../../文件/deepseek百度百科.txt", encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50).split_documents(docs)

embModel = OllamaEmbeddings(model="bge-m3")
vectorDB = Chroma(embedding_function=embModel)
vectorDB.add_documents(chunks)
vectorRetriever = vectorDB.as_retriever()

llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)

multiQueryRetriever = MultiQueryRetriever.from_llm(
    retriever=vectorRetriever,
    llm=llm
)

prompt = ChatPromptTemplate.from_template(
    "根据以下文档回答用户问题:\n"
    "文档内容:\n{doc}"
    "用户提问: {question}"
)

def get_doc_str(data: list[Document]):
    # print(len(data))
    str_docs = []
    for d in data:
        str_docs.append(d.page_content)
    return '\n'.join(str_docs)

chain = (
    RunnablePassthrough.assign(
        doc = RunnableLambda(lambda x: multiQueryRetriever.invoke(x["question"])) | RunnableLambda(get_doc_str)
    )
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke({"question": "deepseek发布了多少个模型"}))

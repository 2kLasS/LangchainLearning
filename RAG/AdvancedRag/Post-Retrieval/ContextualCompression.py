import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter


loader = TextLoader("../../../文件/deepseek百度百科.txt", encoding='utf-8')
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""])

embModel = OllamaEmbeddings(model='bge-m3')
vectorDB = Chroma(embedding_function=embModel)
retriever = vectorDB.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="qwen3-max-2026-01-23", api_key=os.environ.get("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

extractor = LLMChainExtractor.from_llm(llm)
llmFilter = LLMChainFilter.from_llm(llm)


if __name__ == "__main__":
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    vectorDB.add_documents(chunks)

    user_input = "deepseek的发展路径是怎么样的"
    # 从文档中提取出相关的内容的方式
    compressive_retriever = ContextualCompressionRetriever(base_compressor=extractor, base_retriever=retriever)
    print(compressive_retriever.invoke(user_input))

    # 通过LLM过滤掉不相关文档的方式
    compressive_retriever = ContextualCompressionRetriever(base_compressor=llmFilter, base_retriever=retriever)
    print(compressive_retriever.invoke(user_input))
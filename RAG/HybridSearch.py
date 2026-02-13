from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

loader = TextLoader("../文件/deepseek百度百科.txt", encoding='utf-8')
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""])

embModel = OllamaEmbeddings(model='bge-m3')
vectorDB = Chroma(embedding_function=embModel)
vectorRetriever = vectorDB.as_retriever(search_kwargs={"k": 4})


if __name__ == "__main__":
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    vectorDB.add_documents(chunks)

    bm25Retriever = BM25Retriever.from_documents(chunks)
    emsembleRetriever = EnsembleRetriever(retrievers=[bm25Retriever, vectorRetriever], weights=[0.4, 0.6])
    print(emsembleRetriever.invoke("deepseek和360的关系是什么"))
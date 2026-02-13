import os

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_compressors import DashScopeRerank

docs = [
    Document(
        page_content="FastAPI 是一个基于 Starlette 和 Pydantic 的现代 Python Web 框架。它支持异步编程，能够通过 async/await 提升高并发场景下的性能表现。在 I/O 密集型任务中，异步函数通常优于同步函数。",
        metadata={"doc_id": 1}
    ),
    Document(
        page_content="在使用 FastAPI 构建接口时，如果数据库查询较慢，可以考虑使用异步数据库驱动，例如 asyncpg 或 SQLAlchemy 的 async 版本，从而避免阻塞主线程。",
        metadata={"doc_id": 2}
    ),
    Document(
        page_content="RAG 系统通常包括文档切分、向量化、检索和重排序（Rerank）几个步骤。Rerank 的作用是对初步召回的文档进行语义层面的精细排序。",
        metadata={"doc_id": 3}
    ),
    Document(
        page_content="在高并发系统中，线程池和协程是两种不同的并发模型。协程依赖事件循环调度，在 Python 中通常通过 asyncio 实现。",
        metadata={"doc_id": 4}
    ),
    Document(
        page_content="使用 Chroma 作为向量数据库时，可以通过设置 embedding function 和 collection_name 来管理不同的知识库。",
        metadata={"doc_id": 5}
    ),
    Document(
        page_content="Rerank 模型通常使用 Cross-Encoder 架构，它会同时输入 query 和 document，从而计算更精准的相关度分数。",
        metadata={"doc_id": 6}
    ),
    Document(
        page_content="FastAPI 的性能优势不仅来源于异步机制，还得益于其自动生成 OpenAPI 文档的能力以及基于类型注解的高效数据校验。",
        metadata={"doc_id": 7}
    ),
    Document(
        page_content="在 LangChain 中，可以通过 MultiVectorRetriever 或自定义 Retriever 实现多路召回，然后再结合 Rerank 提升结果精度。",
        metadata={"doc_id": 8}
    ),
    Document(
        page_content="如果接口处理的是 CPU 密集型任务，例如图像处理或复杂计算，仅使用 async 并不能显著提升性能，此时可能需要多进程方案。",
        metadata={"doc_id": 9}
    ),
    Document(
        page_content="向量检索的原理是将文本映射到高维向量空间，并通过余弦相似度或欧式距离计算文本之间的相似度。",
        metadata={"doc_id": 10}
    ),
    Document(
        page_content="在构建大规模问答系统时，仅依赖向量相似度排序可能会出现“语义漂移”问题，因此需要额外的重排序模型进行校正。",
        metadata={"doc_id": 11}
    ),
    Document(
        page_content="异步编程适用于 I/O 密集型场景，例如网络请求、文件读取和数据库操作，但不适合 CPU 密集型任务。",
        metadata={"doc_id": 12}
    ),
]

embModel = OllamaEmbeddings(model='bge-m3')
vectorDB = Chroma(embedding_function=embModel)
myRetriever = vectorDB.as_retriever()
reranker = DashScopeRerank(model="qwen3-rerank")

vectorDB.add_documents(docs)

embResult = myRetriever.invoke("如何解决向量检索排序不准确的问题")
print("原始检索")
print(embResult)

rerankResult = reranker.rerank(embResult, "如何解决向量检索排序不准确的问题")
print("重排结果")
print(rerankResult)

def get_rerank_doc(rerank_result, ori_doc):
    doc_list = list()
    for rank in rerank_result:
        doc_list.append(ori_doc[rank['index']])
    return doc_list

final_doc = get_rerank_doc(rerankResult, embResult)
print(final_doc)
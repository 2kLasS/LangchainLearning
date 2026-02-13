import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from langchain_core.documents import Document
from pydantic import BaseModel, Field


def multi_query_retriever(queries: list[str]) -> list[list[Document]]:
    doc_list = []
    for query in queries:
        doc = myRetriever.invoke(query)
        doc_list.append(doc)
    return doc_list


def RRF(retrievalResult: list[list[Document]], k=10, n=3) -> list[Document]:
    mixed_score = {}    # 评分表
    id_to_doc = {}      # 映射表

    for doc_list in retrievalResult:
        for i, doc in enumerate(doc_list):
            if doc.id not in id_to_doc:
                id_to_doc[doc.id] = doc
            if doc.id not in mixed_score:
                mixed_score[doc.id] = 0
            mixed_score[doc.id] += 1 / (k + i)

    sorted_scores = sorted(mixed_score.items(), key=lambda x: x[1], reverse=True)
    return [id_to_doc[did] for did, score in sorted_scores[:n]]


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

llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)

embModel = OllamaEmbeddings(model='bge-m3')
vectorDB = Chroma(embedding_function=embModel)
myRetriever = vectorDB.as_retriever()

generate_query_prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是一个用于优化向量检索效果的查询改写助手。

任务：
基于用户的原始问题，从不同语义维度生成 3 个新的检索问题，
以提高向量数据库的召回覆盖率。

要求：
1. 每个问题必须围绕原始问题核心主题展开，不得偏离主题。
2. 三个问题应分别侧重：
   - 原因或原理角度
   - 解决方案或实现方法角度
   - 相关背景或扩展概念角度
3. 不要只是简单同义改写。
4. 使用中文提问。
    """),
    HumanMessagePromptTemplate.from_template(
        "用户问题:\n{question}"
    )
])

class Question(BaseModel):
    question: list[str] = Field(..., description="基于用户输入重写生成的问题")

generate_query_chain = generate_query_prompt | llm.with_structured_output(Question)


if __name__ == "__main__":
    vectorDB.add_documents(docs)

    user_query = "如何解决向量检索排序不准确的问题"
    generate_result = generate_query_chain.invoke({"question": user_query})
    questions : list[str] = generate_result.question
    questions.append(user_query)

    emb_result = multi_query_retriever(questions)
    rrf_result = RRF(emb_result)
    print(rrf_result)
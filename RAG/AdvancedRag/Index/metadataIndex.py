import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough, RunnableLambda, chain

from langchain_core.documents import Document
from langchain_classic.chains.query_constructor.base import AttributeInfo

"""
首先，系统通过预先构造的查询规划提示词（Query Constructor Prompt）向语言模型说明可用的元数据字段、支持的比较操作以及示例映射方式。该提示词通常包含字段说明、约束条件和少量示例，以引导模型将自然语言问题转换为结构化查询表达。

随后，语言模型根据该提示词生成结构化输出，其中包含两部分内容：一是用于语义检索的文本查询（query），二是以领域专用语言（DSL）形式表示的元数据过滤条件（filter）。该过滤条件描述了字段约束关系，如年份范围、作者匹配或技术领域限定。

接下来，系统使用解析器对模型生成的过滤表达式进行语法分析。解析器基于预定义的语法规则构建抽象语法树（AST），将字符串形式的过滤条件转换为结构化逻辑表达，从而使其能够被程序安全、确定性地处理。

在此基础上，系统将该抽象语法树翻译为向量数据库所支持的过滤语法，并在数据库内部执行元数据筛选。符合条件的文档片段被选为候选集合，随后仅在该集合上执行向量相似度检索，以获取语义相关度最高的结果。

最后，检索得到的文档片段被整理为上下文输入，并注入到回答生成提示词中，由语言模型进行综合阅读与归纳，从而生成最终的自然语言回答。
"""


# 文档内容
docs = [
    Document(
        page_content="作者A团队开发出基于人工智能的自动驾驶决策系统，在复杂路况下的响应速度提升300%",
        metadata={"year": 2024, "rating": 9.2, "genre": "AI", "author": "A"},
    ),
    Document(
        page_content="区块链技术成功应用于跨境贸易结算，作者B主导的项目实现交易确认时间从3天缩短至30分钟",
        metadata={"year": 2023, "rating": 9.8, "genre": "区块链", "author": "B"},
    ),
    Document(
        page_content="云计算平台实现量子计算模拟突破，作者C构建的新型混合云架构支持百万级并发计算",
        metadata={"year": 2022, "rating": 8.6, "genre": "云", "author": "C"},
    ),
    Document(
        page_content="大数据分析预测2024年全球经济趋势，作者A团队构建的模型准确率超92%",
        metadata={"year": 2023, "rating": 8.9, "genre": "大数据", "author": "A"},
    ),
    Document(
        page_content="人工智能病理诊断系统在胃癌筛查中达到三甲医院专家水平，作者B获医疗科技创新奖",
        metadata={"year": 2024, "rating": 7.1, "genre": "AI", "author": "B"},
    ),
    Document(
        page_content="基于区块链的数字身份认证系统落地20省市，作者C设计的新型加密协议通过国家级安全认证",
        metadata={"year": 2022, "rating": 8.7, "genre": "区块链", "author": "C"},
    ),
    Document(
        page_content="云计算资源调度算法重大突破，作者A研发的智能调度器使数据中心能效提升40%",
        metadata={"year": 2023, "rating": 8.5, "genre": "云", "author": "A"},
    ),
    Document(
        page_content="大数据驱动城市交通优化系统上线，作者B团队实现早晚高峰通行效率提升25%",
        metadata={"year": 2024, "rating": 7.4, "genre": "大数据", "author": "B"},
    )
]
# 字段描述
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="文章的领域，选项为['AI'，'区块链'，'云'，'大数据']其中之一",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="文章的出版年份",
        type="integer",
    ),
    AttributeInfo(
        name="author",
        description="署名文章的作者姓名",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="技术价值评估得分（1-10分）",
        type="float"
    )
]


llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)

embModel = OllamaEmbeddings(model="bge-m3")
vectorDB = Chroma(embedding_function=embModel)

retriever = SelfQueryRetriever.from_llm(
    llm = llm,
    vectorstore = vectorDB,
    document_contents = "技术文章简述",
    metadata_field_info=metadata_field_info,
)
retriever.vectorstore.add_documents(docs)


prompt = ChatPromptTemplate.from_template(
    "根据以下文档内容，回答用户的问题，如果文档中没有相关内容，请回复我不知道，如果文档有写，在回答时不要说明你是根据文档回答的，除非用户特别要求"
    "文档内容:\n{doc}"
    "用户提问:\n{query}"
)

@chain
def serialize_documents(data: list[Document]) -> str:
    final_doc = []
    for doc in data:
        str_doc = (f"文章简介:{doc.page_content}\n"
                   f"文章领域:{doc.metadata.get('genre')}\n"
                   f"发表年份:{doc.metadata.get('year')}\n"
                   f"作者:{doc.metadata.get('author')}\n"
                   f"文章评分:{doc.metadata.get('rating')}\n")
        final_doc.append(str_doc)
    return "\n".join(final_doc)

chain = (
    RunnablePassthrough.assign(
        doc=(
            RunnableLambda(lambda x: retriever.invoke(x["query"]))
            | serialize_documents
        )
    )
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke({"query" : "区块链的最新进展是什么"})
print(result)

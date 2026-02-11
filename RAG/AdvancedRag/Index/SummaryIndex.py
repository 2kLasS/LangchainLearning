import os
import uuid

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.stores import InMemoryStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 测试使用在线嵌入模型
# embedding_model = DashScopeEmbeddings(model = "text-embedding-v4")
#
# result = embedding_model.embed_query("真是个潮种")
# print(f"向量化结果:\n{result}")
# print(f"向量化长度: {len(result)}")

# 读取并切分文档
print("正在切分文档...")
origin_doc = TextLoader("../../文件/deepseek百度百科.txt", encoding="utf-8").load()
mySplitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", ""]
)
# 返回一个列表，里面存储的是Document对象
chunks = mySplitter.split_documents(origin_doc)
# print(chunks[:2])
print("切分文档成功")

# 创建数据库存储
print("正在创建数据库...")
embedding_model = OllamaEmbeddings(model="bge-m3")

docStore = InMemoryStore()
vectorStore = Chroma(
    collection_name="ds_information",
    embedding_function=embedding_model
)

# 检索器
idKey = "doc_id"
retriever = MultiVectorRetriever(
    vectorstore=vectorStore,
    docstore=docStore,
    id_key=idKey
)
print("数据库创建成功")

# 文档id列表
doc_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

# 存入原始切块文档
print("正在存储原始文档...")
id_doc_pairs = list(zip(doc_ids, chunks))
retriever.docstore.mset(id_doc_pairs)
print("原始文档存储成功")

# 对文本做摘要
llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)
summarize_prompt = ChatPromptTemplate.from_template("对以下文本进行精简摘要，仅返回摘要文本\n:{text}")

summarize_chain = (
    {"text": lambda x: getattr(x, "page_content", "文档提取失败，里面没有任何内容，请返回无")}
    | summarize_prompt
    | llm
    | StrOutputParser()
)

# try_batch = chunks[:2]
print("正在生成文本摘要...")
summarization = summarize_chain.batch(chunks, config={"max_concurrency" : 5})
print("生成摘要成功")

# 将摘要转为Document对象并加上id
print("正在写入向量数据库...")
summarization = [Document(doc, metadata={idKey: doc_ids[i]}) for i, doc in enumerate(summarization)]
retriever.vectorstore.add_documents(summarization)
print("存入向量数据成功")

# 测试返回结果，是一个Document列表
# result = retriever.invoke("deepseek的公司在哪")
# print(result)
# 返回值
"""
[Document(metadata={'source': '../文件/deepseek百度百科.txt'}, page_content='DeepSeek，全称杭州深度求索人工智能基础技术研究有限公司。
DeepSeek是一家创新型科技公司，成立于2023年7月17日 ，使用数据蒸馏技术，得到更为精炼、有用的数据。由知名私募巨头幻方量化孕育而生，专注于开发先进的大语言模型（LLM）和相关技术。
注册地址：浙江省杭州市拱墅区环城北路169号汇金国际大厦西1幢1201室。法定代表人为裴湉，经营范围包括技术服务、技术开发、软件开发等。'), 
Document(metadata={'source': '../文件/deepseek百度百科.txt'}, page_content='2024年1月5日至6月，
相继发布DeepSeek LLM、DeepSeek-Coder、DeepSeekMath、DeepSeek-VL、DeepSeek-V2、DeepSeek-Coder-V2模型。 
2024年9月5日，更新API支持文档，宣布合并DeepSeek Coder V2和DeepSeek V2 Chat，推出DeepSeek V2.5。
12月13日，发布DeepSeek-VL2。12月26日，正式上线DeepSeek-V3首个版本并同步开源。2025年1月31日，英伟达宣布DeepSeek-R1模型登陆NVIDIANIM。
同一时段内，亚马逊和微软也接入DeepSeek-R1模型。 2月5日，DeepSeek-R1、V3、Coder等系列模型，已陆续上线国家超算互联网平台。
2月6日消息，澳大利亚政府以所谓“担心安全风险”为由，已禁止在所有政府设备中使用DeepSeek。2月8日，DeepSeek正式登陆苏州，并在苏州市公共算力服务平台上完成部署上线，为用户提供开箱即用的软硬件一体服务。 
截至2月9日，DeepSeek App的累计下载量已超1.1亿次，周活跃用户规模最高近9700万')]
"""

def get_content(question: dict) -> str:
    user_input = question.get("question")
    doc_result = retriever.invoke(user_input)

    docs = ""
    for doc in doc_result:
        docs += (f"文档内容：{doc.page_content}\n"
                 f"文档编号：{doc.metadata.get(idKey)}\n\n")
    return docs


rag_prompt = ChatPromptTemplate.from_template("""
根据以下文档内容，回答用户的问题，如果文档中没有提及，请回复我不知道：
文档内容：
{text}

用户问题:
{question}
""")

rag_chain = (
    RunnablePassthrough.assign(
        text = RunnableLambda(get_content)
    ) | rag_prompt | llm | StrOutputParser()
)


while True:
    user_data = input("请输入你的问题，或者输入stop退出: ")
    if user_data == "stop":
        break
    resp = rag_chain.invoke({"question" : user_data})
    print(resp)

print("程序结束")
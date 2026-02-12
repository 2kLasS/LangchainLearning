# 问题分解分为并行分解与串行分解，并行分解只要将问题拆分为多个子问题，再对子问题进行检索，最后将文档汇总，类似于多路召回，比较简单，不再重复实现
# 串行分解需要对上一个子问题的回答，再次生成问题，直到将原始问题解决
# 采用新的代码结构方式，便于理解

import os

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage


def get_str_doc(data: list[Document]):
    str_docs = []
    for d in data:
        str_docs.append(d.page_content)
    return '\n'.join(str_docs)


docs = [
    Document(page_content="阿尔伯特·爱因斯坦（Albert Einstein）于1879年3月14日出生在德国乌尔姆市。他是著名的理论物理学家，提出了相对论。"),
    Document(page_content="德国（Germany）是位于欧洲中部的国家。德国的首都是柏林（Berlin）。德国是欧盟的重要成员国。"),
    Document(page_content="乌尔姆（Ulm）是德国巴登-符腾堡州的一座城市。它位于多瑙河沿岸，以哥特式大教堂闻名。"),
    Document(page_content="柏林是德国的首都和最大城市。柏林是德国的政治、经济和文化中心。"),
    Document(page_content="爱因斯坦是20世纪最伟大的物理学家之一。他因光电效应研究获得1921年诺贝尔物理学奖。"),
]

llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)

embModel = OllamaEmbeddings(model='bge-m3')
vectorDB = Chroma(embedding_function=embModel)
myRetriever = vectorDB.as_retriever()

memory = InMemoryChatMessageHistory()

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个多步推理规划器"),
    HumanMessagePromptTemplate.from_template(
        """
原始问题:
{question}

当前的推理过程:
{history}

请分析问题，并根据当前推理步骤，判断是否需要再对问题进行分解，生成下一个子问题
如果已经足够回答原问题，则不再继续生成，并在返回的JSON字段中返回True

返回规则：
你只能返回JSON格式，形式如下:
{{
    response: str,
    is_end: bool
}}
如果需要问下一个子问题，则返回{{response: "子问题的内容", is_end: False}}
如果不需要下一个子问题，则返回{{response: "", is_end: True}}
        """
    )
])

answer_prompt = ChatPromptTemplate.from_template(
    "根据以下文档内容回答问题\n"
    "文档内容:\n{doc}\n"
    "问题:\n{question}\n"
    "输出简洁的，基于事实的回答。"
)

final_prompt = ChatPromptTemplate.from_template(
    """
原始问题: {question}
推理过程: {history}
基于以上推理过程，生成最终的返回答案。
    """
)

# chain拼接
planner_chain = planner_prompt | llm | JsonOutputParser()

answer_chain = (
    RunnablePassthrough.assign(
        doc = RunnableLambda(lambda x: myRetriever.invoke(x["question"]))
              | RunnableLambda(get_str_doc)
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

final_chain = final_prompt | llm | StrOutputParser()

# 运行逻辑
if __name__ == "__main__":
    vectorDB.add_documents(docs)

    # user_question = "爱因斯坦出生在哪个国家？那个国家的首都是哪？"
    user_question = "爱因斯坦出生的国家的首都的天气怎么样"
    planner_result = planner_chain.invoke({"question": user_question, "history": memory.messages})
    print(f"planner_result:\n{planner_result}")
    memory.add_user_message(HumanMessage(user_question))
    memory.add_ai_message(AIMessage(str(planner_result)))

    is_end = planner_result["is_end"]
    while not is_end:
        answer_result = answer_chain.invoke({"question": planner_result["response"]})
        print(f"answer_result:\n{answer_result}")
        memory.add_ai_message(AIMessage("子问题:"+planner_result["response"]))
        memory.add_ai_message(AIMessage(answer_result))

        planner_result = planner_chain.invoke({"question": user_question, "history": memory.messages})
        print(f"planner_result:\n{planner_result}")
        memory.add_ai_message(AIMessage(str(planner_result)))
        is_end = planner_result["is_end"]

    final_answer = final_chain.invoke({"question": user_question, "history": memory.messages})
    print(final_answer)

# 还有很多优化空间 比如加入最大步数限制，优化记忆结构等
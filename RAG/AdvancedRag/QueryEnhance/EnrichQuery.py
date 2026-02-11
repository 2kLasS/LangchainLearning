import os

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser


llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)

check_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("判断用户输入的问题是否属于编程技术类问题，返回一个JSON，{{is_correct_question:<boolean>, response:<string>}}\n"
                  "如果不是技术类问题，则返回False，并告知用户只能回答编程技术相关的问题"
                  "当问题正确时，response不要回复任何内容"),
    HumanMessagePromptTemplate.from_template("用户问题: {query}")
])

check_chain = check_prompt | llm | JsonOutputParser()

intent_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
"""
你是一个查询理解与重写模块。

你的任务：
1. 判断用户意图 intent
2. 抽取关键参数 params
3. 判断信息是否足够（is_completion）
4. 若不完整，在 response 中向用户提问
5. 若完整，在 response 中生成【高质量向量检索查询语句】
------------------------------------------------
【最终查询语句（response）生成规则 — 极其重要】

当 is_completion = true 时，response 必须满足：

- 用自然语言完整描述用户真实需求
- 不要口语化，不要包含“我想问”“请问”
- 不要包含追问或系统说明
- 不要出现 JSON
- 适合作为知识库搜索语句
- 语义清晰、具体、可匹配文档标题或正文
- 尽量包含：
    - 主题 / 对象
    - 动作（查询什么 / 如何 / 条件）
    - 约束（版本 / 场景 / 错误 / 时间等）

好的示例：
- "FastAPI 中 Depends 的使用方法与依赖注入示例"
- "LangChain 使用 Chroma 构建 RAG 系统的完整流程"
- "订单退款流程与到账时间说明"
- "Windows 环境 Python 3.10 安装步骤"

差的示例（禁止）：
- "怎么弄"
- "帮我看看这个"
- "这个问题"
- "FastAPI 的问题"
- "退款"
------------------------------------------------
【缺失参数时】

- is_completion = false
- response 只用于向用户提问
- 不得生成最终查询语句
------------------------------------------------
【其他规则】

- intent 不要频繁变化，除非用户明确改变需求
- params 必须结构化
- response 语言与用户输入一致
"""
    ),
    MessagesPlaceholder('history'),
    HumanMessagePromptTemplate.from_template("用户输入:\n{query}")
])

intent_chain = intent_prompt | llm | JsonOutputParser()
history = InMemoryChatMessageHistory()


# 处理逻辑
while True:
    user_data = input("请输入你的问题，或者输入stop退出: ")
    if user_data == "stop":
        break

    check_result = check_chain.invoke({"query": user_data})
    if check_result["is_correct_question"]:
        intent = intent_chain.invoke({"query": user_data, "history": history.messages})
        history.add_user_message(HumanMessage(user_data))
        history.add_ai_message(AIMessage(str(intent)))
        print(intent)

        while not intent["is_completion"]:
            user_data = input("LLM追问: ")
            intent = intent_chain.invoke({"query": user_data, "history": history.messages})
            history.add_user_message(HumanMessage(user_data))
            history.add_ai_message(AIMessage(str(intent)))
            print(intent)
        print(f"最终query: {intent['response']}")
        history.clear()
    else:
        print(check_result["response"])

"""
拿到改写的输出后，再将问题用去向量数据库检索进行RAG...
"""
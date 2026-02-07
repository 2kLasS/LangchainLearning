import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory

# 第一种写法，手动维护记忆
# prompt = ChatPromptTemplate.from_messages([
#     SystemMessage("你是一个人工智能助手"),
#     MessagesPlaceholder(variable_name="history"),
# ])
#
# llm = ChatOpenAI(model="qwen3-max-2026-01-23", api_key=os.environ.get("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
#
# memory = InMemoryChatMessageHistory()
#
# chain = RunnablePassthrough.assign(
#     history = lambda _: memory.messages,
# ) | prompt | llm | StrOutputParser()
#
# while True:
#     user_input = input("请输入你的问题，输入stop退出: ")
#     if user_input == "stop":
#         break
#     memory.add_message(HumanMessage(user_input))
#     full = ""
#     resp = chain.stream({})
#     for chunk in resp:
#         full += chunk
#         print(chunk, end="", flush=True)
#     print()
#     memory.add_message(AIMessage(full))
# print("Have a nice day!")


# 第二种写法，自动维护
prompt = ChatPromptTemplate.from_messages([
    SystemMessage("你是一个人工智能助手"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{question}")
])

llm = ChatOpenAI(model="qwen3-max-2026-01-23", api_key=os.environ.get("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

memory = InMemoryChatMessageHistory()


def get_memory():
    return memory

base_chain = prompt | llm | StrOutputParser()

auto_memory_chain = RunnableWithMessageHistory(
    base_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="history"
)

while True:
    user_input = input("请输入你的问题，输入stop退出: ").strip()
    if user_input == "stop":
        break
    resp = auto_memory_chain.stream({"question" : user_input})
    for chunk in resp:
        print(chunk, end="", flush=True)
    print()
print("END")
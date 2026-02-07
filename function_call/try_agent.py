import os
import datetime

from langchain_core.messages import HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def get_time():
    """获取今天的日期，返回格式年-月-日"""
    return datetime.date.today().strftime("%Y-%m-%d")

@tool
def get_nearby_store():
    """获取附近的奶茶店和便利店有哪些"""
    return """
    附近两百米有蜜雪冰城、瑞幸咖啡、一点点、茶百道等奶茶品牌
    同时有711、美宜佳便利店
    """


llm = ChatOpenAI(model="qwen3-max-2026-01-23",
                 api_key=os.environ.get("DASHSCOPE_API_KEY"),
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

Tools = [get_time, get_nearby_store]

agent = create_agent(
    model=llm,
    tools=Tools,
)

prompt = HumanMessagePromptTemplate.from_template("{text}")

while True:
    user_input = input("输入你所需要的帮助: ")
    if user_input == "stop":
        break
    final_prompt = prompt.format(text=user_input)
    resp = agent.invoke({"messages": [final_prompt]})
    print(resp)
    print(resp["messages"][-1].content)
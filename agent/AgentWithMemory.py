import os
import numexpr
import math

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool


@tool
def calculator(expression: str):
    """
    使用 numexpr 计算单行数学表达式，可用的常量符号e，pi。
    例如：
      - "37593 * 67"
      - "37593**(1/5)"
      - "(2 + 3) * 4"
    """

    local_dict = {'e': math.e, 'pi': math.pi}
    global_dict = {}

    result = str(numexpr.evaluate(expression, local_dict, global_dict))
    return result


llm = ChatOpenAI(model="qwen3-max-2026-01-23",
                 api_key=os.environ.get("DASHSCOPE_API_KEY"),
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


agent = create_agent(model=llm, tools=[calculator], checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "user_001"}}

response1 = agent.invoke({"messages": [{"role": "user", "content": "5*20是多少"}]}, config=config)
print(response1)

response2 = agent.invoke({"messages": [{"role": "user", "content": "再把他开平方是多少"}]}, config=config)
print(response2)
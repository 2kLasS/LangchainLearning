import os
import math
import numexpr as ne

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from pydantic import BaseModel, Field


@tool
def calculator(exp: str):
    """
    使用 numexpr 计算单行数学表达式，可用的常量符号e，pi。
    例如：
      - "37593 * 67"
      - "37593**(1/5)"
      - "(2 + 3) * 4"
    """
    local_dict = {'e': math.e, 'pi': math.pi}
    global_dict = {}

    result = str(ne.evaluate(exp, local_dict, global_dict))
    return result


class Schedule(BaseModel):
    plans: list[str] = Field(..., description="指定的计划列表")


llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

llm_with_search = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_search": True}
)

planner = create_agent(
    model=llm,
    response_format=ToolStrategy(Schedule),
    system_prompt="你是一个规划师，需要将收到的任务拆分成多个小任务"
)

executor = create_agent(
    model=llm_with_search,
    tools=[calculator],
    checkpointer=InMemorySaver(),
)


if __name__ == "__main__":
    planner_result = planner.invoke({"messages": [{"role": "user", "content": "一万块钱能买多少黄金"}]})
    config = {"configurable": {"thread_id": "user_001"}}
    for step, task in enumerate(planner_result["structured_response"].plans):
        print(f"正在执行第{step+1}个任务: {task}")
        resp = executor.invoke({"messages": [{"role": "user", "content": task}]}, config=config)
        print(resp)
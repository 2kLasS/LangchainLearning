import os
import datetime

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


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


tools = [get_time, get_nearby_store]
tool_map = {t.name: t for t in tools}

llm = ChatOpenAI(model="qwen3-max-2026-01-23", api_key=os.environ.get("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

with_tool_llm = llm.bind_tools(tools)

history = InMemoryChatMessageHistory()

# 手动处理工具调用
def chat(user_input, max_range=5):
    history.add_message(HumanMessage(user_input))
    # 不超过最大步骤
    for _ in range(max_range):
        resp = with_tool_llm.invoke(history.messages)
        history.add_message(resp)
        tc = resp.tool_calls

        # 如果没有工具调用，说明是最终答案
        if not tc:
            return resp

        for tool in tc:
            selected_tool = tool_map[tool["name"]]
            result = selected_tool.invoke(tool["args"])
            history.add_message(ToolMessage(
                    content=str(result),
                    tool_call_id=tool["id"],
                )
            )


    return "超过了最大尝试次数，也许问题太复杂了，我们换个问题试试吧"


print(chat("今天几号了，我不记得我上次喝奶茶是啥时候了，而且我想喝点甜的"))

import os
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

class Book(BaseModel):
    title: str = Field(..., description="书名")
    author: str = Field(..., description="作者名")
    category: str = Field(..., description="书籍分类")
    summary: str = Field(..., description="书籍的大概内容")

llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)
llm = llm.with_structured_output(Book)

chain = llm | RunnableLambda(lambda x: x.model_dump_json(indent=2))

resp = chain.invoke("告诉我《像艺术家一样思考》这本书的信息")
print(resp)
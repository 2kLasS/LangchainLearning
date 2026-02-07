import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="qwen3-max-2026-01-23", api_key=os.environ.get("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

prompt = ChatPromptTemplate.from_template("{question}")

chain = prompt | llm | StrOutputParser()

resp = chain.stream({"question" : "你好，你是谁"})
for chunk in resp:
    print(chunk, end="", flush=True)

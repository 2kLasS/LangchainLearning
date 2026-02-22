import os
import pymysql

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool

conn = pymysql.connect(
    host="127.0.0.1",
    user=os.environ.get("db_user"),
    password=os.environ.get("db_password"),
    database="scoredb",
)
cursor = conn.cursor()

@tool
def get_database_information()->str:
    """获取当前数据库中的表与表结构的信息"""
    db_info = []

    cursor.execute("show tables")
    table_names = cursor.fetchall()
    for table in table_names:
        cursor.execute(f"show create table {table[0]}")
        result = cursor.fetchall()
        table_info = f"表名: {result[0][0]}\n建表语句: {result[0][1]}"
        db_info.append(table_info)
    return "\n".join(db_info)

@tool
def execute_sql(sql_statement:str)->tuple:
    "执行sql语句并返回执行结果"
    # 此处可以优化，应当加入错误处理，且有sql注入的风险
    cursor.execute(sql_statement)
    result = cursor.fetchall()
    return result


llm = ChatOpenAI(model="qwen3-max-2026-01-23",
                 api_key=os.environ.get("DASHSCOPE_API_KEY"),
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

agent = create_agent(llm, [get_database_information, execute_sql])


if __name__ == "__main__":
    resp = agent.invoke({"messages": [{"role": "user", "content": "看看高等数学有没有人考100分，有的话给我他的详细信息"}]})
    print(resp)

    cursor.close()
    conn.close()
